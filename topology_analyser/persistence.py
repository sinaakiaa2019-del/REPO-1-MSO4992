"""TDA: daily persistence diagrams, Betti curves and compact features."""

from config import ROOT, HERE, OUT
import json
import numpy as np
import pandas as pd
import hashlib, concurrent.futures, time
from matrix_builder import correlation
from ripser import ripser
from scipy.special import ndtr
from scipy.linalg import eigh

from .tda_config import GRID
from .betti_curves import landscape_top1, network_summary
from .wasserstein import w2


def checksum(path):
    """Return the SHA-256 checksum used to validate a checkpoint."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def features(c):
    """Measure connected components and loops over one fixed distance grid."""
    dist = np.sqrt(np.maximum(0, 2 * (1 - c)))
    np.fill_diagonal(dist, 0)
    d = ripser(dist, distance_matrix=True, maxdim=1, thresh=2)["dgms"]
    h0 = d[0][np.isfinite(d[0][:, 1])]
    h1 = d[1]
    assert np.isfinite(h1).all()
    curves = np.stack(
        [((g[:, 0, None] <= GRID) & (GRID < g[:, 1, None])).sum(axis=0) for g in d], axis=-1
    )
    life = np.diff(h1).ravel()
    n = len(c)
    prob = life / life.sum() if life.sum() > 0 else np.array([])
    summary = dict(
        n_assets=n,
        h1_entropy_normalized=float(-(prob * np.log(prob)).sum() / np.log(max(len(life), 2))),
        h1_max=float(life.max()) if len(life) else 0.0,
        h1_total_per_asset=float(life.sum() / n),
        h1_count_per_asset=len(life) / n,
        h1_integrated_l2_per_asset=float(np.sqrt(np.trapezoid(curves[:, 1] ** 2, GRID)) / n),
        h0_integrated_l2_per_asset=float(np.sqrt(np.trapezoid(curves[:, 0] ** 2, GRID)) / n),
        h0_mean_lifetime=float(np.diff(h0).mean()) if len(h0) else 0,
        average_correlation=float((c.sum() - n) / (n * (n - 1))),
    )
    summary["eigenvalue_ratio"] = float(
        eigh(c, subset_by_index=[n - 1, n - 1], eigvals_only=True)[0] / n
    )
    landscape1 = landscape_top1(h1)
    summary["landscape1_l1_per_asset"] = float(np.trapezoid(landscape1, GRID) / n)
    summary["landscape1_l2_per_asset"] = float(np.sqrt(np.trapezoid(landscape1**2, GRID)) / n)
    summary["network_clustering"], summary["network_algebraic_connectivity"] = network_summary(c)
    edges = np.linspace(0, 2, 21)
    image = np.zeros((20, 20))
    for birth, death in h1:
        lifetime = death - birth
        image += lifetime * np.outer(
            np.diff(ndtr((edges - birth) / 0.05)), np.diff(ndtr((edges - lifetime) / 0.05))
        )
    return (h0, h1, curves, summary, image)


def compute(job):
    """Calculate persistent-homology features for one rolling window."""
    day, a, path = job
    c = correlation(a)
    h0, h1, curves, s, pi = features(c)
    s["date"] = day
    # Publish a complete checkpoint atomically, so interruptions cannot leave
    # a partial file that the next run mistakes for a finished window.
    temporary = path.with_suffix(".pending")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, h0=h0, h1=h1, betti=curves, summary=json.dumps(s), image=pi)
    temporary.replace(path)
    return day


def research_window(job):
    """One fully observed member window for the declared sensitivity grid."""
    day, block, lam, shrink, path = job
    h0, h1, curves, summary, _ = features(correlation(block, lam, shrink))
    summary["date"] = day
    temporary = path.with_suffix(".pending")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, h1=h1, betti=curves, summary=json.dumps(summary))
    temporary.replace(path)
    return day


def research_topology(spec, destination):
    """Recompute daily TDA with no liquidity cap and dated membership."""
    destination.mkdir(parents=True, exist_ok=True)
    returns = pd.read_pickle(HERE / "returns.pkl")
    if spec.get("mask_extreme"):
        # A sensitivity analysis, not a claim that large genuine losses are errors.
        returns = returns.where(returns.abs() <= 0.5)
    members = (pd.read_csv(HERE / "membership_fja.csv", parse_dates=["date"])
               .set_index("date").tickers.sort_index().reindex(returns.index, method="ffill"))
    colmap = {s.replace(".", "-"): i for i, s in enumerate(returns.columns)}
    values = returns.to_numpy()
    window = spec["window"]
    pending = set()
    paths = []
    with concurrent.futures.ProcessPoolExecutor(4) as pool:
        for t in range(window, len(returns)):
            block = values[t-window+1:t+1]
            valid = np.isfinite(block).all(axis=0) & (np.std(block, axis=0) > 1e-12)
            eligible = [colmap[s.replace(".", "-")] for s in sorted(set(members.iloc[t].split(",")))
                        if s.replace(".", "-") in colmap and valid[colmap[s.replace(".", "-")]]]
            if len(eligible) < 3:
                raise ValueError("Too few observed members on " + str(returns.index[t]))
            day = str(returns.index[t].date())
            path = destination / (day + ".npz")
            paths.append(path)
            if not path.exists():
                pending.add(pool.submit(research_window, (day, block[:, eligible], spec["lambda"], spec["shrink"], path)))
            if len(pending) >= 8:
                done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    future.result()
            if t % 500 == 0:
                print("Research TDA", spec["name"], t, "/", len(returns), flush=True)
        for future in concurrent.futures.as_completed(pending):
            future.result()
    summaries, diagrams, betti = [], [], []
    for path in paths:
        with np.load(path) as z:
            summaries.append(json.loads(str(z["summary"])))
            diagrams.append(z["h1"])
            betti.append(z["betti"][:, 1])
    result = pd.DataFrame(summaries).set_index("date")
    result.index = pd.to_datetime(result.index)
    with concurrent.futures.ProcessPoolExecutor(4) as pool:
        velocity = [np.nan] + list(pool.map(w2, diagrams[:-1], diagrams[1:], chunksize=20))
    result["w2_per_sqrt_asset"] = np.asarray(velocity) / np.sqrt(result.n_assets.rolling(2).mean())
    for j in [80, 90, 100, 110]:
        result[f"betti_{j:03}"] = np.asarray(betti)[:, j] / result.n_assets.to_numpy()
    return result


def run_topology():
    """Process every eligible historical member each day; resume hashed windows."""
    protocol = json.loads((ROOT / "config.json").read_text())
    inputs = [
        ROOT / "cache/prices.pkl",
        ROOT / "cache/markets.pkl",
        ROOT / "cache/membership_fja.csv",
        ROOT / "config.json",
        *sorted(ROOT.glob("*.py")),
        *sorted((ROOT / "topology_analyser").glob("*.py")),
    ]
    hashes = {str(p.relative_to(ROOT)): checksum(p) for p in inputs}
    from importlib.metadata import version

    runtime = {name: version(name) for name in ["numpy", "pandas", "scipy", "ripser"]}
    fp = hashlib.sha256(json.dumps([hashes, runtime], sort_keys=True).encode()).hexdigest()[:16]
    cache = HERE / "windows" / fp
    cache.mkdir(parents=True, exist_ok=True)
    (OUT / "input_manifest.json").write_text(
        json.dumps(
            dict(fingerprint=fp, hashes=hashes, runtime=runtime, protocol=protocol), indent=2
        )
    )
    spy = pd.read_pickle(ROOT / "cache/markets.pkl").SPY.dropna()
    p = pd.read_pickle(ROOT / "cache/prices.pkl").reindex(spy.index)
    assert p.index.is_unique and p.columns.is_unique
    returns = np.log(p.where(p > 0)).diff()
    returns.to_pickle(HERE / "returns.pkl")
    member = (
        pd.read_csv(ROOT / "cache/membership_fja.csv", parse_dates=["date"])
        .set_index("date")
        .tickers.sort_index()
        .reindex(p.index, method="ffill")
    )
    normalized = [s.replace(".", "-") for s in p.columns]
    assert len(set(normalized)) == len(normalized)
    colmap = {s: i for i, s in enumerate(normalized)}
    a = returns.to_numpy()
    rows = []
    ledger = []
    pending = set()
    count = 0
    started = time.time()
    with concurrent.futures.ProcessPoolExecutor(4) as pool:
        for t in range(60, len(p)):
            names = sorted(set(member.iloc[t].split(",")))
            mapped = [colmap[s.replace(".", "-")] for s in names if s.replace(".", "-") in colmap]
            block = a[t - 59 : t + 1]
            valid = np.isfinite(block).all(axis=0) & (np.std(block, axis=0) > 1e-12)
            eligible = [i for i in mapped if valid[i]]
            day = str(p.index[t].date())
            missing = sorted((s for s in names if s.replace(".", "-") not in colmap))
            rows.append(
                dict(
                    date=day,
                    expected_members=len(names),
                    mapped=len(mapped),
                    eligible=len(eligible),
                    missing_symbols=len(missing),
                    excluded_incomplete_or_constant=len(mapped) - len(eligible),
                )
            )
            ledger.extend(
                (dict(date=day, symbol=s, reason="unresolved price symbol") for s in missing)
            )
            ledger.extend(
                (
                    dict(
                        date=day,
                        symbol=p.columns[i],
                        reason="incomplete 60-session returns or constant",
                    )
                    for i in mapped
                    if not valid[i]
                )
            )
            if len(eligible) < 3:
                continue
            path = cache / f"{day}.npz"
            if path.exists():
                continue
            pending.add(pool.submit(compute, (day, block[:, eligible], path)))
            if len(pending) >= 8:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for f in done:
                    f.result()
                    count += 1
                    if count % 250 == 0:
                        print(
                            "Daily windows",
                            count,
                            "elapsed",
                            round(time.time() - started),
                            flush=True,
                        )
        for f in concurrent.futures.as_completed(pending):
            f.result()
    pd.DataFrame(rows).to_csv(OUT / "coverage.csv", index=False)
    pd.DataFrame(ledger).to_csv(OUT / "exclusion_ledger.csv", index=False)
    summaries = []
    curves = []
    images = []
    diagrams = []
    for path in sorted(cache.glob("*.npz")):
        with np.load(path) as z:
            summaries.append(json.loads(str(z["summary"])))
            curves.append(z["betti"])
            images.append(z["image"])
            diagrams.append(z["h1"])
    summary = pd.DataFrame(summaries).set_index("date")
    summary.index = pd.to_datetime(summary.index)
    with concurrent.futures.ProcessPoolExecutor(4) as pool:
        velocities = [np.nan]
        for i, x in enumerate(pool.map(w2, diagrams[:-1], diagrams[1:], chunksize=10)):
            velocities.append(x)
            if i % 1000 == 0:
                print("W2 windows", i, flush=True)
    summary["w2_per_sqrt_asset"] = np.array(velocities) / np.sqrt(
        summary.n_assets.rolling(2).mean()
    )
    summary.to_csv(OUT / "topology_features.csv")
    summary.to_pickle(HERE / "topology.pkl")
    np.savez_compressed(
        OUT / "betti_curves.npz",
        betti=np.stack(curves),
        epsilon=GRID,
        dates=summary.index.to_numpy(dtype="datetime64[D]"),
    )
    np.savez_compressed(
        OUT / "persistence_images.npz",
        images=np.stack(images),
        dates=summary.index.to_numpy(dtype="datetime64[D]"),
    )
    print("TOPOLOGY COMPLETE", len(summary), flush=True)

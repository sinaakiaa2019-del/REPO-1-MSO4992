"""Data processing: price-quality checks and historical asset coverage."""

from config import ROOT, HERE, OUT
from pathlib import Path
import json
import numpy as np
import pandas as pd


def audit_reference_prices():
    """Check a documented historical close against the original supplied row.

    This check flags a security-history problem. It does not substitute one
    isolated quote into an unverified adjusted-price series.
    """
    contract = json.loads((OUT / "real_input_contract.json").read_text())
    raw = Path(
        contract.get(
            "price_path",
            str(Path(contract.get("input_directory", ROOT / "inputs")) / "prices.csv"),
        )
    )
    unavailable = dict(
        symbol="CFC",
        date="2007-11-14",
        field="close",
        supplied_close=None,
        reference_close=13.37,
        absolute_difference=None,
        status="not_available",
        source="https://www.sec.gov/Archives/edgar/data/25191/000095012407005885/v35644orsv3asr.htm",
        interpretation="The raw close reference is not present in this input; no match or mismatch asserted.",
    )
    if not raw.exists() or "close" not in pd.read_csv(raw, nrows=0).columns:
        return pd.DataFrame([unavailable])
    found = []
    for chunk in pd.read_csv(
        raw, usecols=["date", "symbol", "close", "adjusted_close"], chunksize=250000
    ):
        row = chunk[(chunk.symbol == "CFC") & (chunk.date == "2007-11-14")]
        if len(row):
            found.append(row)
    if not found:
        return pd.DataFrame([unavailable])
    row = pd.concat(found).iloc[0]
    return pd.DataFrame(
        [
            dict(
                symbol="CFC",
                date="2007-11-14",
                field="close",
                supplied_close=float(row.close),
                reference_close=13.37,
                absolute_difference=abs(float(row.close) - 13.37),
                status="mismatch" if abs(float(row.close) - 13.37) > 0.02 else "match",
                source="https://www.sec.gov/Archives/edgar/data/25191/000095012407005885/v35644orsv3asr.htm",
                interpretation="An isolated raw-close check; adjusted-return history remains unverified. No price replacement performed.",
            )
        ]
    )


def audit_inputs():
    """Keep price-quality and unresolved-identity evidence for the full file."""
    contract = OUT / "real_input_contract.json"
    raw = (
        Path(
            json.loads(contract.read_text()).get(
                "price_path",
                str(
                    Path(json.loads(contract.read_text())["input_directory"])
                    / "prices.csv"
                ),
            )
        )
        if contract.exists()
        else ROOT / "cache/execution_bundle/TDA_Crash_Project/inputs/prices.csv"
    )
    d = pd.read_csv(raw, usecols=["date", "symbol", "adjusted_close"])
    assert not d.duplicated(["date", "symbol"]).any()
    invalid = ~np.isfinite(d.adjusted_close) | d.adjusted_close.le(0)
    d.loc[invalid].to_csv(OUT / "invalid_prices.csv", index=False)
    d.loc[invalid, "adjusted_close"] = np.nan
    p = d.pivot(index="date", columns="symbol", values="adjusted_close").sort_index()
    p.index = pd.to_datetime(p.index)
    prior = pd.read_pickle(ROOT / "cache/prices.pkl")
    pd.testing.assert_frame_equal(p, prior, check_dtype=False, check_index_type=False)
    markets = pd.read_pickle(ROOT / "cache/markets.pkl")
    calendar = markets.SPY.dropna().index
    r = np.log(p.reindex(calendar)).diff()
    extreme = r.stack()
    extreme = extreme[extreme.abs() > 0.5]
    extreme.rename("log_return").to_csv(OUT / "extreme_returns_review.csv")
    membership = pd.read_csv(ROOT / "cache/membership_fja.csv", parse_dates=["date"])
    assert not membership.date.duplicated().any()
    snapshots = (
        membership.set_index("date")
        .tickers.sort_index()
        .reindex(r.index, method="ffill")
    )
    member_sets = {
        day: {s.replace(".", "-") for s in names.split(",")}
        for day, names in snapshots.items()
    }
    extreme_frame = extreme.rename("log_return").reset_index()
    extreme_frame.columns = ["date", "symbol", "log_return"]
    extreme_frame["member_on_observation_date"] = [
        symbol.replace(".", "-") in member_sets[day]
        for day, symbol in zip(extreme_frame.date, extreme_frame.symbol)
    ]
    extreme_frame.to_csv(OUT / "extreme_returns_membership_review.csv", index=False)
    intervals = []
    active = {}
    for row in membership.sort_values("date").itertuples():
        names = set(row.tickers.split(","))
        for ticker in set(active) - names:
            intervals.append(
                dict(symbol=ticker, start=active.pop(ticker), end_exclusive=row.date)
            )
        for ticker in names - set(active):
            active[ticker] = row.date
    intervals.extend(
        (dict(symbol=t, start=s, end_exclusive=pd.NaT) for t, s in active.items())
    )
    pd.DataFrame(intervals).to_csv(OUT / "membership_intervals.csv", index=False)
    current_path = ROOT / "cache/sectors_current.csv"
    current = (
        pd.read_csv(current_path)
        if current_path.exists()
        else pd.DataFrame(columns=["Symbol", "CIK"])
    )
    current.Symbol = current.Symbol.str.replace(".", "-", regex=False)
    current = current.set_index("Symbol")
    master = pd.DataFrame({"price_symbol": p.columns})
    master["normalized_symbol"] = master.price_symbol.str.replace(".", "-", regex=False)
    master["current_CIK_reference"] = master.normalized_symbol.map(current.CIK)
    master["identity_status"] = (
        "ticker/punctuation match only; historical security identity unverified"
    )
    master["historical_sector_status"] = "unavailable"
    master.to_csv(OUT / "security_master_unresolved.csv", index=False)
    audit = dict(
        rows=len(d),
        symbols=int(d.symbol.nunique()),
        invalid_nonpositive_prices=int(invalid.sum()),
        extreme_returns_flagged=len(extreme),
        duplicates=0,
        raw_matches_previous_panel=True,
        spy_sessions=len(calendar),
        original_price_dates=len(p),
        price_dates_not_SPY_sessions=[
            str(x.date()) for x in p.index.difference(calendar)
        ],
        permanent_security_ids="unresolved",
        historical_sector_classifications="unavailable",
    )
    (OUT / "data_audit.json").write_text(json.dumps(audit, indent=2))
    audit["extreme_returns_while_index_member"] = int(
        extreme_frame.member_on_observation_date.sum()
    )
    (OUT / "data_audit.json").write_text(json.dumps(audit, indent=2))
    statuses = [
        dict(
            requirement="Full historical sector leadership",
            status="blocked_by_missing_dated_classifications",
            action="No current-survivor result reported as historical",
        ),
        dict(
            requirement="Full foreign/credit constituent transfer",
            status="blocked_by_missing_target_panels",
            action="Small observed baskets evaluated separately as exploratory",
        ),
        dict(
            requirement="Permanent security identity resolution",
            status="unresolved",
            action="No guessed ticker renaming; unresolved mapping ledger supplied",
        ),
        dict(
            requirement="Untouched final holdout",
            status="unavailable",
            action="All rerun scores labelled retrospective development evidence",
        ),
    ]
    pd.DataFrame(statuses).to_csv(OUT / "external_data_gaps.csv", index=False)
    references = audit_reference_prices()
    references.to_csv(OUT / "reference_price_checks.csv", index=False)
    if references.status.eq("mismatch").any():
        print(
            "DATA QUALITY: a historical raw close disagrees with its documented reference. See reference_price_checks.csv; empirical conclusions remain provisional."
        )
    print(json.dumps(audit, indent=2))


def audit_universe():
    """Account for every supplied ticker, including former and unused members.

    The end-of-period group is only a diagnostic. It never selects training
    assets. Leaving the index is not treated as proof that a company delisted.
    """
    returns = pd.read_pickle(HERE / "returns.pkl")
    prices = pd.read_pickle(HERE / "prices.pkl").reindex(returns.index)
    snapshots = pd.read_csv(HERE / "membership_fja.csv", parse_dates=["date"])
    names = (
        snapshots.set_index("date")
        .tickers.sort_index()
        .reindex(returns.index, method="ffill")
    )
    sets = [{s.replace(".", "-") for s in value.split(",")} for value in names]
    all_members = set().union(*sets)
    supplied = {s.replace(".", "-"): s for s in prices.columns}
    complete = returns.notna().rolling(60).sum().eq(60)
    # Compute each window directly: rolling updates can retain rounding residue
    # after a formerly variable series becomes exactly constant.
    values = returns.to_numpy()
    varying = pd.DataFrame(False, index=returns.index, columns=returns.columns)
    for end in range(60, len(returns)):
        varying.iloc[end] = np.std(values[end - 59 : end + 1], axis=0) > 1e-12
    final_members = sets[-1]
    rows = []
    daily = pd.DataFrame(index=returns.index)
    daily["expected_members"] = [len(s) for s in sets]
    daily["matched_members"] = 0
    daily["eligible_members"] = 0
    daily["eligible_former_members"] = 0
    for normalized in sorted(all_members | set(supplied)):
        member = pd.Series([normalized in day for day in sets], index=returns.index)
        symbol = supplied.get(normalized)
        eligible = (
            member & complete[symbol] & varying[symbol] if symbol else member & False
        )
        if symbol:
            daily["matched_members"] += member.astype(int)
        daily["eligible_members"] += eligible.astype(int)
        if normalized not in final_members:
            daily["eligible_former_members"] += eligible.astype(int)
        valid_prices = prices[symbol].dropna() if symbol else pd.Series(dtype=float)
        overlap = int((member & prices[symbol].notna()).sum()) if symbol else 0
        rows.append(
            dict(
                symbol=symbol or normalized,
                supplied=symbol is not None,
                membership_sessions=int(member.sum()),
                first_membership_date=(
                    str(member[member].index.min().date()) if member.any() else None
                ),
                last_membership_date=(
                    str(member[member].index.max().date()) if member.any() else None
                ),
                valid_prices_during_membership=overlap,
                eligible_sessions=int(eligible.sum()),
                first_valid_price=(
                    str(valid_prices.index.min().date()) if len(valid_prices) else None
                ),
                last_valid_price=(
                    str(valid_prices.index.max().date()) if len(valid_prices) else None
                ),
                member_at_end=normalized in final_members,
                status=(
                    "used"
                    if eligible.any()
                    else (
                        "no dated membership match"
                        if not member.any()
                        else (
                            "missing price symbol"
                            if not symbol
                            else (
                                "no valid prices during membership"
                                if overlap == 0
                                else "no complete valid window"
                            )
                        )
                    )
                ),
                identity="historical security identity unresolved",
            )
        )
    ledger = pd.DataFrame(rows)
    ledger.to_csv(OUT / "asset_coverage.csv", index=False)
    daily.to_csv(OUT / "survivorship_daily.csv", index_label="date")
    actual = pd.read_csv(OUT / "coverage.csv", index_col=0, parse_dates=True)
    np.testing.assert_array_equal(
        daily.loc[actual.index, "eligible_members"], actual.eligible
    )
    summary = dict(
        supplied_symbols=len(supplied),
        symbols_ever_in_membership=len(all_members),
        supplied_symbols_used=int(
            (ledger.supplied & ledger.eligible_sessions.gt(0)).sum()
        ),
        supplied_symbols_never_used=int(
            (ledger.supplied & ledger.eligible_sessions.eq(0)).sum()
        ),
        missing_historical_price_symbols=int((~ledger.supplied).sum()),
        supplied_symbols_with_no_membership_price_overlap=int(
            (
                ledger.supplied
                & ledger.membership_sessions.gt(0)
                & ledger.valid_prices_during_membership.eq(0)
            ).sum()
        ),
        former_member_symbols_used=int(
            (~ledger.member_at_end & ledger.eligible_sessions.gt(0)).sum()
        ),
        former_member_asset_sessions=int(daily.eligible_former_members.sum()),
        expected_member_sessions=int(daily.expected_members.sum()),
        matched_member_sessions=int(daily.matched_members.sum()),
        eligible_member_sessions_after_warmup=int(actual.eligible.sum()),
        expected_member_sessions_after_warmup=int(actual.expected_members.sum()),
        survivorship_bias_eliminated=False,
        conclusion="Dated membership reduces survivor selection; missing histories, ticker identities and terminal delisting returns remain unresolved.",
    )
    (OUT / "survivorship_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

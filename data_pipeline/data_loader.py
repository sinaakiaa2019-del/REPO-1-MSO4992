"""Data inputs: validate real files and record their source checksums."""

import sys
from pathlib import Path

if __name__ == "__main__":
    # Running this file directly (e.g. `python data_pipeline/data_loader.py
    # --refresh-public ...`) only puts this folder on sys.path, not the
    # project root, so config.py would not be found without this line.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT, HERE, OUT
import os
import json
import numpy as np
import pandas as pd
import shutil, hashlib


def sha(path):
    """Return the SHA-256 checksum of a local input file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def prepare(folder, prices_file=None):
    """Validate real inputs and preserve invalid prices as missing observations."""
    folder = Path(folder).resolve()
    work = HERE
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    required = [
        "membership.csv",
        "markets.csv",
        "vix.csv",
        "yield_curve.csv",
        "source_manifest.json",
    ]
    price_path = Path(prices_file).resolve() if prices_file else folder / "prices.csv"
    missing = [name for name in required if not (folder / name).is_file()]
    if not price_path.is_file():
        missing.append(str(price_path))
    if missing:
        raise FileNotFoundError("Required real inputs missing: " + ", ".join(missing))
    provenance = json.loads((folder / "source_manifest.json").read_text())
    if provenance.get("data_kind") != "real":
        raise ValueError("Real entry point requires a real-data source manifest")
    d = pd.read_csv(price_path, usecols=["date", "symbol", "adjusted_close"])
    d.date = pd.to_datetime(d.date, errors="raise")
    if d[["date", "symbol"]].isna().any().any() or d.duplicated(["date", "symbol"]).any():
        raise ValueError("Missing or duplicate date/security keys")
    if d.symbol.astype(str).str.match("^SYN\\d{3,}$").any():
        raise ValueError("Synthetic generator symbols in real mode")
    d.adjusted_close = pd.to_numeric(d.adjusted_close, errors="raise")
    bad = ~np.isfinite(d.adjusted_close) | (d.adjusted_close <= 0)
    d.loc[bad].to_csv(out / "invalid_input_prices.csv", index=False)
    d.loc[bad, "adjusted_close"] = np.nan
    prices = d.pivot(index="date", columns="symbol", values="adjusted_close").sort_index()
    markets = pd.read_csv(folder / "markets.csv", index_col=0, parse_dates=True).sort_index()
    if markets.index.min() < pd.Timestamp("2000-01-01") or markets.index.max() > pd.Timestamp(
        "2024-12-31"
    ):
        raise ValueError(
            "This experiment is locked to 2000–2024; revise the protocol for a later study"
        )
    if "SPY" not in markets or markets.index.has_duplicates or markets.SPY.dropna().le(0).any():
        raise ValueError("Invalid SPY price/calendar contract")
    member = pd.read_csv(folder / "membership.csv", parse_dates=["date"])
    if member.date.duplicated().any() or member.tickers.isna().any():
        raise ValueError("Ambiguous membership snapshots")
    if any(len(names.split(",")) != len(set(names.split(","))) for names in member.tickers):
        raise ValueError("Duplicate symbols within a membership snapshot")
    if member.date.min() > markets.SPY.dropna().index.min():
        raise ValueError("Membership starts after modelling calendar")
    vix = pd.read_csv(folder / "vix.csv")
    if pd.to_datetime(vix.DATE).duplicated().any():
        raise ValueError("Duplicate VIX dates")
    pd.to_datetime(vix.DATE, errors="raise")
    pd.to_numeric(vix.CLOSE, errors="raise")
    slope = pd.read_csv(folder / "yield_curve.csv", index_col=0, parse_dates=True)
    if slope.index.has_duplicates:
        raise ValueError("Duplicate yield-curve dates")
    if list(slope.columns) != ["T10Y2Y"]:
        raise ValueError("Yield curve must be actual T10Y2Y, not a yield-level proxy")
    prices.to_pickle(work / "prices.pkl")
    markets.to_pickle(work / "markets.pkl")
    for source, target in [
        ("membership.csv", "membership_fja.csv"),
        ("vix.csv", "vix.csv"),
        ("yield_curve.csv", "t10y2y_real.csv"),
    ]:
        shutil.copy2(folder / source, work / target)
    (out / "real_input_contract.json").write_text(
        json.dumps(
            {
                "data_kind": "real",
                "input_directory": str(folder),
                "price_path": str(price_path),
                "sources": provenance,
                "hashes": {**{n: sha(folder / n) for n in required}, "prices.csv": sha(price_path)},
                "rows": len(d),
                "symbols": prices.shape[1],
                "invalid_prices": int(bad.sum()),
            },
            indent=2,
        )
    )
    print("Real input contracts passed", len(d), "rows", prices.shape[1], "symbols", flush=True)


def refresh_public(original_prices):
    """Fetch the full historical symbol union; retain failures and raw responses.

    The public feed is not a permanent-security-ID or delisting-return service.
    No unavailable symbol is filled from another input or silently substituted.
    """
    import concurrent.futures
    import datetime
    import sqlite3
    import urllib.request
    import urllib.error
    import urllib.parse
    import time
    import zlib

    folder = ROOT / 'inputs'
    member = pd.read_csv(folder / 'membership.csv')
    symbols = {s.replace('.', '-') for names in member.tickers for s in names.split(',')}
    for chunk in pd.read_csv(original_prices, usecols=['symbol'], chunksize=250000):
        symbols.update(chunk.symbol.dropna().str.replace('.', '-', regex=False))
    sectors = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
    archive = HERE / 'public_source.sqlite'
    with sqlite3.connect(archive) as con:
        con.execute('CREATE TABLE IF NOT EXISTS responses (symbol TEXT PRIMARY KEY, retrieved_at TEXT, raw BLOB)')

    def fetch(symbol):
        """Download and validate one requested equity price history."""
        url = 'https://query1.finance.yahoo.com/v8/finance/chart/' + urllib.parse.quote(symbol, safe='')
        url += '?period1=946684800&period2=1735689600&interval=1d&events=div%2Csplits'
        record = dict(symbol=symbol, url=url, status='failed', rows=0,
                      retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            with sqlite3.connect(archive) as con:
                saved = con.execute('SELECT retrieved_at, raw FROM responses WHERE symbol=?', (symbol,)).fetchone()
            if saved:
                record['retrieved_at'], raw = saved[0], zlib.decompress(saved[1])
            else:
                for attempt in range(2):
                    try:
                        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, timeout=25) as response:
                            raw = response.read()
                        break
                    except urllib.error.HTTPError as error:
                        if error.code == 429 and attempt == 0:
                            time.sleep(min(30, max(2, int(error.headers.get('Retry-After', '5')))))
                        else:
                            raise
                with sqlite3.connect(archive, timeout=60) as con:
                    con.execute('INSERT OR REPLACE INTO responses VALUES (?,?,?)',
                                (symbol, record['retrieved_at'], zlib.compress(raw)))
            record['response_sha256'] = hashlib.sha256(raw).hexdigest()
            response = json.loads(raw)['chart']['result'][0]
            meta = response['meta']
            for key in ['symbol', 'longName', 'shortName', 'instrumentType', 'exchangeName', 'currency', 'firstTradeDate']:
                record['provider_' + key] = meta.get(key)
            if meta['symbol'].upper().replace('.', '-') != symbol.upper():
                raise ValueError('Provider resolved to a different symbol; no rename assumed')
            if symbol not in sectors and meta.get('instrumentType') != 'EQUITY':
                raise ValueError('Constituent request did not resolve to an equity')
            if symbol in sectors and meta.get('instrumentType') != 'ETF':
                raise ValueError('Sector request did not resolve to an ETF')
            dates = pd.to_datetime(response['timestamp'], unit='s', utc=True)
            dates = dates.tz_convert(meta['exchangeTimezoneName']).tz_localize(None).normalize()
            adjusted = response['indicators']['adjclose'][0]['adjclose']
            frame = pd.DataFrame(dict(date=dates, symbol=symbol, adjusted_close=adjusted))
            frame = frame[frame.date.between('2000-01-01', '2024-12-31')].copy()
            if frame.date.duplicated().any():
                raise ValueError('Duplicate observation dates')
            bad = ~np.isfinite(frame.adjusted_close) | frame.adjusted_close.le(0)
            record['invalid_observations'] = int(bad.sum())
            frame.loc[bad, 'adjusted_close'] = np.nan
            if not frame.adjusted_close.notna().any():
                raise ValueError('No positive adjusted prices in requested period')
            record.update(status='available', rows=len(frame), first_date=str(frame.date.min().date()),
                          last_date=str(frame.date.max().date()),
                          identity_status='Ticker metadata only; permanent historical identity unverified')
            actions=[]
            for kind, entries in response.get('events', {}).items():
                for item in entries.values():
                    actions.append(dict(symbol=symbol, kind=kind, details=json.dumps(item, sort_keys=True)))
            return record, frame, actions
        except Exception as error:
            record['detail'] = type(error).__name__ + ': ' + str(error)[:250]
            return record, None, []

    frames, fund_frames, audit, actions = [], [], [], []
    requested = sectors + sorted(symbols - set(sectors))
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        for i, (record, frame, events) in enumerate(pool.map(fetch, requested), 1):
            audit.append(record)
            actions.extend(events)
            if frame is not None:
                (fund_frames if record['symbol'] in sectors else frames).append(frame)
            if i % 50 == 0:
                pd.DataFrame(audit).to_csv(folder / 'public_download_audit.csv', index=False)
                print('Public quotes', i, '/', len(requested), 'available', len(frames), flush=True)
    pd.DataFrame(audit).to_csv(folder / 'public_download_audit.csv', index=False)
    pd.DataFrame(actions).to_csv(folder / 'corporate_actions.csv', index=False)
    if len(frames) < 300 or len(fund_frames) != 9:
        raise RuntimeError('Insufficient real public data; inspect public_download_audit.csv')
    prices = pd.concat(frames).sort_values(['date', 'symbol'])
    prices.to_csv(folder / 'public_prices.csv', index=False)
    funds = pd.concat(fund_frames).pivot(index='date', columns='symbol', values='adjusted_close').sort_index()
    funds.to_csv(folder / 'sector_etfs.csv')
    source = json.loads((folder / 'source_manifest.json').read_text())
    source.update(equity_prices='Fresh public Yahoo chart observations for full original/membership symbol union',
                  price_sha256=sha(folder / 'public_prices.csv'),
                  cleaning_status='Finite-positive observations only; no future-neighbour filter or ticker exclusion list',
                  retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  public_source_limit='Ticker identity, delisting coverage and historical adjustment vintages unverified',
                  requested_symbols=len(symbols), available_equity_symbols=len(frames),
                  sector_scope='Nine pre-2000 Select Sector ETFs; evolving fund exposures, not historical firm-sector labels')
    source.pop('cleaned_file_received', None)
    (folder / 'source_manifest.json').write_text(json.dumps(source, indent=2))
    print('PUBLIC REFRESH COMPLETE', len(prices), 'rows', len(frames), 'equity histories; 9 sector funds', flush=True)


def refresh_wiki_recovery(missing_symbols_file, api_key=None):
    """Recover delisted-equity history from Nasdaq Data Link's free WIKI/PRICES dataset.

    Covers only symbols the public refresh above could not find (results/asset_coverage.csv,
    status == 'missing price symbol'). WIKI/PRICES is community-maintained and frozen at
    2018-03-27, so it cannot help with 2019-2024; rows outside 2000-2024 are dropped.
    Writes only inputs/wiki_recovered_prices.csv and inputs/wiki_recovery_audit.csv --
    nothing in results/ or the existing inputs/public_prices.csv is touched.
    """
    import csv
    import urllib.request
    import urllib.error
    import urllib.parse
    import time

    api_key = api_key or os.environ.get('NDL_API_KEY')
    if not api_key:
        raise SystemExit('Set NDL_API_KEY or pass api_key=')
    start, end, delay = '2000-01-01', '2018-03-27', 1.3

    def fetch(ticker):
        """Download and validate one supporting market series."""
        url = ('https://data.nasdaq.com/api/v3/datatables/WIKI/PRICES'
               f'?ticker={urllib.parse.quote(ticker)}&api_key={api_key}'
               f'&date.gte={start}&date.lte={end}&qopts.columns=ticker,date,adj_close')
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as response:
            return json.loads(response.read())

    with open(missing_symbols_file, encoding='utf-8') as f:
        symbols = [line.strip() for line in f if line.strip()]

    rows, audit = [], []
    for i, symbol in enumerate(symbols, 1):
        ticker = symbol.replace('.', '_')  # WIKI convention for share classes
        try:
            data = fetch(ticker).get('datatable', {}).get('data', [])
            if not data:
                audit.append(dict(symbol=symbol, status='no_data', rows=0))
            else:
                for _, date, adj_close in data:
                    if adj_close is not None and adj_close > 0:
                        rows.append((date, symbol, adj_close))
                audit.append(dict(symbol=symbol, status='recovered', rows=len(data)))
        except urllib.error.HTTPError as error:
            audit.append(dict(symbol=symbol, status=f'http_{error.code}', rows=0))
        except Exception as error:
            audit.append(dict(symbol=symbol, status=type(error).__name__, rows=0))
        if i % 25 == 0:
            print(f'{i}/{len(symbols)} processed, {len(rows)} price rows so far', flush=True)
        time.sleep(delay)

    with open(ROOT / 'inputs/wiki_recovered_prices.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'symbol', 'adjusted_close'])
        writer.writerows(sorted(rows))
    with open(ROOT / 'inputs/wiki_recovery_audit.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['symbol', 'status', 'rows'])
        writer.writeheader()
        writer.writerows(audit)

    recovered = sum(1 for a in audit if a['status'] == 'recovered')
    print(f'DONE. {recovered}/{len(symbols)} symbols recovered, {len(rows)} total price rows.', flush=True)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Refresh audited public data for the historical symbol union.')
    parser.add_argument('--refresh-public', type=Path, help='Original equity CSV, used only to enumerate symbols')
    parser.add_argument('--refresh-wiki-recovery', type=Path, help='Text file, one missing symbol per line')
    args = parser.parse_args()
    if args.refresh_wiki_recovery:
        refresh_wiki_recovery(args.refresh_wiki_recovery)
    elif args.refresh_public:
        refresh_public(args.refresh_public)
    else:
        parser.error('Supply --refresh-public or --refresh-wiki-recovery')

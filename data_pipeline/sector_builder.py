"""Historical sectors: record the evidence needed before making sector claims."""

from config import ROOT, HERE, OUT
import json
import numpy as np
import pandas as pd


SECTORS = dict(XLB='Materials', XLE='Energy', XLF='Financials', XLI='Industrials',
               XLK='Technology', XLP='Consumer staples', XLU='Utilities',
               XLV='Health care', XLY='Consumer discretionary')


def delay_summary(values):
    """Shape of trailing return dynamics, not a cross-stock sector network.

    Five consecutive returns form each point. Scalar within-window centring
    and standardisation remove location and volatility before Euclidean PH.
    No future observation is used. The filtration units differ from the
    correlation-distance filtration in the main stock-panel experiment.
    """
    from ripser import ripser
    if not np.isfinite(values).all() or np.std(values) < 1e-12:
        return np.nan, np.nan
    cloud = np.lib.stride_tricks.sliding_window_view(values, 5)
    cloud = (cloud - np.mean(values)) / np.std(values) / np.sqrt(5)
    diagram = ripser(cloud, maxdim=1)['dgms'][1]
    life = np.diff(diagram, axis=1).ravel()
    life = life[np.isfinite(life) & (life > 0)]
    prob = life / life.sum() if len(life) else np.array([])
    entropy = float(-np.sum(prob*np.log(prob))/np.log(max(2,len(life))))
    return float(life.sum()/len(cloud)), entropy


def sector_frame(job):
    """Build the feature frame for one sector-fund experiment."""
    symbol, price, target = job
    from model_training import evaluate_groups
    r = np.log(price.where(price > 0)).diff()
    shape = [delay_summary(r.iloc[t-59:t+1].to_numpy()) for t in range(60,len(r))]
    frame = pd.DataFrame(shape,index=r.index[60:],columns=['delay_total','delay_entropy'])
    frame['vol20']=r.rolling(20).std()*np.sqrt(252)
    frame['momentum20']=np.log(price/price.shift(20))
    frame=frame.join(target)
    groups=dict(sector_market=['vol20','momentum20'],
                sector_topology=['delay_total','delay_entropy'],
                sector_combined=['vol20','momentum20','delay_total','delay_entropy'])
    result=evaluate_groups(frame,groups,budgets=(0.05,))
    result.pop('models')
    for name, value in result.items():
        value['symbol']=symbol
        value['sector_proxy']=SECTORS[symbol]
    print('Sector proxy complete',symbol,flush=True)
    return result


def run_sectors():
    """Test all nine pre-2000 sector funds with an explicit proxy scope."""
    import concurrent.futures
    path=ROOT/'inputs/sector_etfs.csv'
    if not path.exists():
        raise FileNotFoundError('RQ5 requires observed sector_etfs.csv; no synthetic or classification fallback')
    prices=pd.read_csv(path,index_col=0,parse_dates=True)
    spy=pd.read_pickle(HERE/'markets.pkl').SPY.dropna()
    prices=prices.reindex(spy.index)
    forward=pd.concat([spy.shift(-i) for i in range(1,21)],axis=1)
    target=pd.DataFrame(dict(label=(forward.min(axis=1)/spy-1<=-.1).astype(float),
                             label_end=pd.Series(spy.index,index=spy.index).shift(-20)))
    target.loc[target.index[-20:],'label']=np.nan
    jobs=[(s,prices[s],target) for s in SECTORS]
    with concurrent.futures.ProcessPoolExecutor(3) as pool:
        results=list(pool.map(sector_frame,jobs))
    for name in results[0]:
        frame=pd.concat([r[name] for r in results],ignore_index=True)
        if 'columns' in frame:frame['columns']=frame['columns'].map(json.dumps)
        frame.to_csv(OUT/f'rq5_{name}.csv',index=False)
    pd.DataFrame([dict(research_question='RQ5',status='available_sector_ETF_proxy',
        reason='Nine actual sector funds; original historical constituent-sector question remains unresolved')]).to_csv(OUT/'rq5_status.csv',index=False)

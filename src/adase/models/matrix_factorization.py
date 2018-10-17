import numpy as np
import pandas as pd

import networkx as nx
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.base import clone

from .trmf import TRMFRegressor

import matplotlib.pyplot as plt

import time, gzip, pickle, os
from sklearn.model_selection import train_test_split

import tqdm
from sklearn.model_selection import ParameterGrid
from joblib import delayed, Parallel


def matrix_factorization(df_demand: pd.DataFrame, df_topoout: pd.DataFrame) -> pd.DataFrame:

    def smape(y_true, y_pred):
        weight = (np.abs(y_true) + np.abs(y_pred)) / 2
        output = np.divide(np.abs(y_true - y_pred), weight, where=weight > 0,
                           out=np.full_like(weight, np.nan))

        return output  # np.nanmean(output, axis=1)

    df_demand = df_demand.set_index(["item_code", "rpd", "future_flag"]).sort_index()

    df_topoout = df_topoout.set_index(["item_code", "rpd", "future_flag", "parent_item_code"]).sort_index()

    demand_total = df_demand["quantity"].unstack("future_flag", fill_value=0)

    # `periods=-1` shifts the values so that value at `j` becomes the value@`j-1`
    demand_delta = demand_total - demand_total.shift(periods=-1, axis=1).fillna(0.)

    edges = {}
    for (itm, rpd, f_f, par), (qty, par_qty) in tqdm.tqdm(df_topoout.iterrows()):
        edges.setdefault((rpd, f_f), []).append((itm, par, {"qty": qty, "par": par_qty}))

    master = nx.from_edgelist(((u, v) for lst in edges.values() for u, v, _ in lst), create_using=nx.DiGraph)

    # index -> vertex
    itov = dict(enumerate(master))

    # vertex -> index
    vtoi = dict(zip(itov.values(), itov.keys()))

    assert all(i == vtoi[itov[i]] for i in itov)
    assert all(v == itov[vtoi[v]] for v in vtoi)

    transformer = Pipeline([
        ("log1p", FunctionTransformer(func=np.log1p,
                                      inverse_func=np.exp,
                                      validate=False)),  # , check_inverse=False)),
        ("scale", RobustScaler())
    ])

    trmf_base = TRMFRegressor(
        n_components=None,  # do not specify the number of components just yet
        n_order=None,  # do not specify the order of the latent autoregression
        adj=nx.adjacency_matrix(master),  # use the adjacency matrix from the master relation graph
        C_B=0,  # there will be no exogenous features
        fit_regression=False,  # no regression on exogenous features
        fit_intercept=True,  # fit the intercept data
        nonnegative_factors=False,  # do not impose nonnegativity on the internal factor loadings
        n_max_mf_iter=1,  # the number of internal factorization iterations
                            # TODO change to 5
        random_state=0x07C0FFEE,  # fix the random seed (for replayable initialization)
    )

    grid = ParameterGrid({
        "C_Z": [1e-2, 1e0],
        "C_F": [1e-2, 1e0],
        "C_phi": [1e-3, 1e-1],
        "eta_Z": [0.95, 0.99],
        "eta_F": [0.25, 0.75],
        "n_components": [10, 100],
        "n_order": [3, 9],
    })

    par_ = Parallel(n_jobs=-1, verbose=10)

    def helper(i, par, X, base=trmf_base):
        return i, clone(base).set_params(**par).fit(X)

    import time, gzip, pickle, os
    from sklearn.model_selection import train_test_split

    dttm = time.strftime("%Y%m%d%H%M%S")
    print(f"the timestamp is {dttm}")

    for future_flag in range(4):
        # Get the data to run the experiment on
        data = demand_total[future_flag].unstack("item_code", fill_value=0.0)

        # align the columns to the : item_code -> natual number + sort
        full = data.rename(columns=vtoi).sort_index(axis=1)

        # Train / test split
        train, test = train_test_split(full, test_size=8, shuffle=False)

        # To remap the integer-index columns to `item_code` apply the following:
        # ```python
        # # natual number -> item_code + orignal item_code order
        # test.rename(columns=itov).reindex(columns=data.columns)
        # ```

        # Preproces the train
        train_scaled = transformer.fit_transform(train)

        # Run the full experiment
        results = par_(delayed(helper)(i, par, train_scaled)
                       for i, par in enumerate(grid))

        # save in a cache
        cache = f"models/results_{dttm}_ff{future_flag}.gz"
        with gzip.open(cache, "w", compresslevel=6) as fout:
            pickle.dump((results, transformer, test), fout)
        # end with
    # end for

    n_ahead, n_horizon = len(test), 12

    best_results = {}
    for ff in range(4):
        cached_results = f"models/results_{dttm}_ff{ff}.gz"
        with gzip.open(cached_results, "r") as fin:
            results, transformer, test = pickle.load(fin)

        # Collect the forecasts
        test_forecasts = []
        for i, est in tqdm.tqdm(results):
            pred_scaled = est.predict(n_ahead=n_ahead + n_horizon)
            pred = transformer.inverse_transform(pred_scaled)
            test_forecasts.append((i, pred))

        # Compute smape metrics.
        predictions = dict(
            (i, pd.DataFrame(predicted[:n_ahead], index=test.index, columns=test.columns) \
             .rename(columns=itov).reindex(data.columns, axis=1))
            for i, predicted in tqdm.tqdm(test_forecasts))

        metrics_detail = [
            (i, pd.DataFrame(smape(test, predicted[:n_ahead]),
                             index=test.index, columns=test.columns))
            for i, predicted in tqdm.tqdm(test_forecasts)
        ]

        # Collect into a series
        details = pd.concat(dict(
            (i, df.median(axis=1),)
            for i, df in tqdm.tqdm(metrics_detail)
        ), axis=0)

        # chose the least median smape
        df = details.unstack("rpd").median(axis=1)  # .max(axis=1)
        best_par_idx = df.idxmin()
        best_results[ff] = dict(
            par=grid[best_par_idx],
            smape=df.loc[best_par_idx],
            prediction=predictions[best_par_idx]
        )

        # details = pd.concat(dict(
        #     (tuple(par[k] for k in keys), df.median(axis=1),)
        #     for par, df in tqdm.tqdm(metrics_detail)
        # ), axis=0, names=keys)

        # chose the least maximal smape
        # df = details.unstack("rpd").max(axis=1)
        # best_par = dict(zip(keys, df.idxmin()))

    df_submission = pd.concat({
        ff: res["prediction"]
        for ff, res in best_results.items()
    }, axis=0, names=["future_flag"])

    df_submission = df_submission.stack().rename("quantity")

    df_submission = df_submission.reorder_levels(["item_code", "future_flag", "rpd"])
    df_submission = df_submission.sort_index()

    df_submission.to_csv(f"../../../output/huawei_{dttm}_trmf_graph.csv",
                         index=True, header=True)

    return df_submission

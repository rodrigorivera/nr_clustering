import pandas as pd
import pyflux as pf


def arimax(dataset: pd.DataFrame, target: str):
    return pf.ARIMAX(
        data=dataset,
        formula=target+'~1',
        ar=2,
        ma=3,
        family=pf.Normal()
    )
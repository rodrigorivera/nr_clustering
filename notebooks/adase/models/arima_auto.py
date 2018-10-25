import pyramid as pm


def arima_auto(dataset, start_p=1,
               start_q=1, max_p=3,
               max_q=3, m=12,
               start_P=0, seasonal=False,
               d=1, D=1,
               trace=False,
               error_action='ignore',
               suppress_warnings=True,
               stepwise=True):

    return pm.auto_arima(
        dataset,
        start_p=start_p,
        start_q=start_q,
        max_p=max_p,
        max_q=max_q,
        m=m,
        start_P=start_P,
        seasonal=seasonal,
        d=d,
        D=D,
        trace=trace,
        error_action=error_action,
        suppress_warnings=suppress_warnings,
        stepwise=stepwise
    )
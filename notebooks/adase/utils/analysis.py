import os

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import scipy.stats as stats

from .loss_functions import smape, rmsle, differentiable_smape
from .loss_functions import rounded_smape, rmsle2, kaggle_smape, mae


def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x = plt.xticks(rotation=90)


def heatmap(df: pd.DataFrame, title: str):
    sns.set(font_scale=2)
    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(50, 50))
    sns.heatmap(df, annot=False, ax=ax, cmap="YlGnBu", fmt="d")
    plt.title(str(title))
    plt.show()


def anova(frame, qualitative):
    anv = pd.DataFrame()
    anv['feature'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['quantity'].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')


def second_neighbors(graph, node):
    """
    a generator that yeilds 2preprocess neighbors of node in graph
    neighbors are not not unique!
    """
    for neighbor_list in [graph.neighbors(n) for n in graph.neighbors(node)]:
        for n in neighbor_list:
            yield n


def plot_predictions(p_id, ff, target, train_data, test_data, predicted_values, naive_forecast, loss_functions, model):
    fig = plt.figure(1)
    # set up subplot grid
    gridspec.GridSpec(3, 3)
    plt.figure(figsize=(20, 12))
    fig.tight_layout()
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 18}
    plt.rc('font', **font)

    # plot predictions and expected results
    plt.subplot2grid((2, 2), (0, 0))
    plt.scatter([x for x in test_data], [x for x in predicted_values], marker='x', s=100, label=model)
    plt.scatter([x for x in test_data], [x for x in naive_forecast], marker='o', s=100, label='Naive Forecast')
    plt.xlabel('Actual value')
    plt.ylabel('Predicted value')
    plt.legend(loc='best')
    plt.suptitle('Product {}'.format(p_id) + ' - Future Flag {}'.format(ff))

    # plot actuals, naive and prediction
    plt.subplot2grid((2, 2), (0, 1))
    plt.plot(target)
    plt.plot([None for i in train_data] + [x for x in test_data], label='Actual')
    plt.plot([None for i in train_data] + [x for x in predicted_values], label=model)
    plt.plot(naive_forecast, label='Naive Forecast')
    plt.ylabel('Z-score Quantity')
    plt.xlabel('Period')
    plt.legend(loc='best')

    # plot loss functions
    plt.subplot2grid((2, 2), (1, 0), colspan=2, rowspan=1)
    plt.bar(range(len(loss_functions)), list(loss_functions.values()), align='center')
    plt.xticks(range(len(loss_functions)), list(loss_functions.keys()))
    plt.ylabel('Error')
    plt.xlabel('Metric')
    plt.legend(loc='best')

    fig.show()


def prepare_plotting_data(df_predictions, df_train, target, product_id, future_flag):
    df_temp = df_predictions.copy()
    conditions = (df_temp['item_code'] == product_id) & (df_temp['future_flag'] == future_flag) & (
                df_temp['target'] == target)
    sub_df = df_temp[conditions].sort_values(by='SMAPE', ascending=True).groupby(['model']).first().reset_index()

    min_smape = df_temp.groupby(['model'])['SMAPE'].min().values
    pred_df = sub_df[sub_df['SMAPE'] == min_smape].sort_values(by='SMAPE', ascending=False).groupby(
        'SMAPE').first().reset_index()
    pred = sub_df.iloc[0]['y_pred_format']
    # pred = pred_df['y_pred_format'].squeeze()
    # print(pred)

    nd = df_train.copy()
    nd = nd[(nd['item_code'] == product_id) & (nd['future_flag'] == future_flag)].reset_index()
    train = nd[nd['rpd'] < 37][target].squeeze()
    actual = nd[nd['rpd'] > 36][target].squeeze()
    # print(actual)
    naive_target = 'qty_0mean_unitvar_naive_fct'
    # naive_target = 'quantity'
    naive = nd[nd['rpd'] > 36][naive_target].squeeze()
    model = sub_df['model'].values[0]
    target = nd[target]

    losses = {
        'SMAPE': smape(actual, pred) / 200,
        'SMAPE\nDiff': differentiable_smape(actual, pred),
        'SMAPE\nRounded': rounded_smape(actual, pred),
        'SMAPE\nKaggle': kaggle_smape(actual, pred),
        'RMSLE': rmsle(actual, pred),
        'RMSLE\n2preprocess': rmsle2(actual, pred),
        'RMSLE\n3forecast': np.sqrt(mean_squared_log_error(np.abs(actual), np.abs(pred))),
        'MAE': mae(actual, pred),
        'RMSE': np.sqrt(mean_squared_error(actual, pred)),
        'SMAPE\nNaive': smape(actual, naive) / 200,
        'RMSLE\nnaive': rmsle(actual, naive),
        'SMAPE\nMin': min_smape[0] / 200
    }

    return target, train, actual, pred, naive, losses, model
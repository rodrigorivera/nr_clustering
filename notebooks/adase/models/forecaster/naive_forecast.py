import warnings
from typing import List

from src import BuildFeatures

warnings.filterwarnings('ignore')

'''
def _naive_forecast(
                    df: pd.DataFrame,
                    group_condition: List[str],
                    targets: List[str],
                    cfg: Dict
                    ) -> pd.DataFrame:
    data: pd.DataFrame = df.copy()
    data = data.sort_values(by=group_condition + ['rpd'], ascending=True)

    for target in targets:
        logging.debug('Forecast: ' + target)
        data[target + '_naive_fct'] = data.groupby(group_condition)[target].shift(1)
        logging.debug('Done forecast:' + target)
        path = cfg['forecast']['path_sources_processing']
        write_df(data, path, 'naivefct')

    return data


def _create_naive_forecast(cfg: Dict,
                           group_condition: List[str],
                           targets: List[str]
                           ) -> List[pd.DataFrame]:

        path = cfg['global']['path_full'] + cfg['forecast']['path_forecast']
        files = has_files(path)
        dfs_forecast: List[pd.DataFrame] = []

        if files:
            list_of_files = get_all_files(path)
            dfs = read_df([file for file in list_of_files])
            dfs_forecast = dfs

        if not files:

            temp_data = PreProcess()

            for idx, data in enumerate(temp_data.data):
                logging.debug('Ready to naive forecast')
                df_temp = _naive_forecast(data, group_condition, targets, cfg)
                path = cfg['forecast']['path_sources_naive_forecast']
                write_df(df_temp, path, 'final_fct_{}'.format(idx))
                dfs_forecast.append(df_temp)
                logging.debug('Finished forecasting')

        return dfs_forecast
'''


class NaiveForecast(BuildFeatures):

    def __init__(self,
                 group_condition: List[str],
                 targets: List[str],
                 **kwargs
                 ) -> None:
        '''

        :param group_condition: List[str]
        :param targets: List[str]
        :param kwargs:

        This also needs to be refactored to function with scikit pipeline. The result is an additional
        column containing the naive forecast
        '''

        BuildFeatures.__init__(self)

        cfg = self.config()
        config_file = cfg['forecast']

        if kwargs.get('log'):
            self.log(config_file)

        #self.data = _create_naive_forecast(cfg, group_condition, targets)




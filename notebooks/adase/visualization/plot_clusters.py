import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import time

plt.switch_backend('agg')


def plot_clusters(df: pd.DataFrame,
                  target: str,
                  resolution: int,
                  path_images: str,
                  method: str
                  ) -> None:

    for cluster in df['cluster_id'].unique():
        image_title_part = path_images + time.strftime("%Y%m%d_%H%M") + '_huawei_'
        image_title = image_title_part + method + '{}'.format(cluster) + '_' + target + '_{}'.format(
            resolution) + 'dpi.png'

        condition = (df['cluster_id'] == cluster)
        df_sub_series_data = df[condition]

        cluster_target = np.concatenate(np.array(df_sub_series_data[target])).reshape(-1, 45)
        cluster_centroid = np.concatenate(np.array(df_sub_series_data['centroids'])).reshape(-1, 45)

        targets_array = np.concatenate(np.array(df[target])).reshape(-1, 45)
        centroids_array = np.concatenate(np.array(df['centroids'])).reshape(-1, 45)

        idx = np.random.choice(df_sub_series_data['index'].values, 1)[0]
        product_value = df_sub_series_data[df_sub_series_data['index'] == idx]['item_code_future_flag'].values
        centroid_value = df_sub_series_data[df_sub_series_data['index'] == idx]['cluster_id'].values

        plt.figure(1)
        plt.figure(figsize=(20, 10))
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 22}

        plt.rc('font', **font)

        plt.subplot(121)
        plt.plot(targets_array[idx], label='Quantity')
        plt.plot(centroids_array[idx], label='Centroid')
        plt.ylabel('Z-Score Quantity')
        plt.xlabel('Period')
        plt.title('Product {}'.format(product_value))
        plt.legend(loc='best')

        plt.subplot(122)
        plt.plot(cluster_target)
        plt.plot(cluster_centroid)
        plt.ylabel('Z-Score Quantity')
        plt.xlabel('Period')
        plt.title('Cluster: {}'.format(centroid_value) + ' - Shape: {}'.format(targets_array.shape))

        plt.savefig(image_title, dpi=resolution)
        # plt.show()
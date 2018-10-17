import logging
import numpy as np


def _DTWDistance(s1, s2, w) -> float:
    DTW = {}
    w = max(w, abs(len(s1) - len(s2)))

    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])


def _LB_Keogh(s1, s2, r):
    LB_sum = 0

    for ind, i in enumerate(s1):
        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

        if i > upper_bound:
            LB_sum = LB_sum + (i - upper_bound) ** 2
        elif i < lower_bound:
            LB_sum = LB_sum + (i - lower_bound) ** 2

    return np.sqrt(LB_sum)


def create_clusters_kmeans(data, num_clust, num_iter, w=5):
    centroids = data[np.random.choice(data.shape[0], num_clust, replace=False)]
    counter = 0

    for n in range(num_iter):
        counter += 1
        logging.debug(counter)
        assignments = {}
        # assign data points to clusters

        for ind, i in enumerate(data):
            min_dist = float('inf')
            closest_clust = None

            for c_ind, j in enumerate(centroids):
                if _LB_Keogh(i, j, 5) < min_dist:
                    cur_dist = _DTWDistance(i, j, w)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind

            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []

        # recalculate centroids of clusters
        for key in assignments:
            clust_sum = 0

            for k in assignments[key]:
                clust_sum = clust_sum + data[k]

            centroids[key] = [m / len(assignments[key]) for m in clust_sum]

    return centroids


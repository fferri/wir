#!/usr/bin/env python3
print(__doc__)

verbose = False

import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from agglomerate import AgglomerativeClustering

import math
import itertools
from collections import defaultdict
import operator

np.random.seed(0)

if verbose: print('*** Generating datasets...')
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
n_samples = 1500//14
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

plt.figure(figsize=(9, 9.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.92, wspace=.1,
                    hspace=.25)

plot_num = 1

def predict(X, alg):
    algorithm.fit(X)
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)
    return y_pred

def agglomerate(X, ys):
    if len(ys) == 1: return ys[0]
    ac = AgglomerativeClustering(X)
    for y in ys:
        ac.add_clustering(y)
    return ac.compute()

def filter_clustering_noise(X, y, min_cluster_size=5):
    items, count = np.unique(y, return_counts=True)
    mask = np.in1d(y, items[count >= min_cluster_size])
    return X[mask], y[mask]

datasets = [noisy_circles, noisy_moons, blobs, no_structure]
dataset_name = ['noisy_circles', 'noisy_moons', 'blobs', 'no_structure']
for i_dataset, dataset in enumerate(datasets):
    print('*** Working on dataset "%s"...' % dataset_name[i_dataset])
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # create clustering estimators
    clustering_algorithms = defaultdict(dict)
    for k in (2, 3, 4, 5):
        clustering_algorithms['MiniBatchKMeans'][k] = cluster.MiniBatchKMeans(n_clusters=k)
        clustering_algorithms['SpectralClustering'][k] = cluster.SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity="nearest_neighbors")
        clustering_algorithms['Ward'][k] = cluster.AgglomerativeClustering(n_clusters=k, linkage='ward', connectivity=connectivity)
        clustering_algorithms['AgglomerativeClustering'][k] = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=k, connectivity=connectivity)
        clustering_algorithms['Birch'][k] = cluster.Birch(n_clusters=k)
    for eps in (0.01, 0.05, 0.2, 0.5):
        clustering_algorithms['DBSCAN[eps=%.2f]'%eps][0] = cluster.DBSCAN(eps=eps)
    clustering_algorithms['MeanShift'][0] = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    for damping in (0.6, 0.9):
        clustering_algorithms['AffinityPropagation[damp=%.1f]'%damping][0] = cluster.AffinityPropagation(damping=damping, preference=-200)

    scores = defaultdict(dict)
    y_pred_all = {}
    y_centers = {}

    agglom = {}

    for name, algorithms in clustering_algorithms.items():
        if True or verbose: print('    *** algorithm %s k values: %s...' % (name, tuple(algorithms.keys())))
        ys = []
        for k, algorithm in algorithms.items():
            if True or verbose: print('    *** running algorithm %s[k=%d]...' % (name, k))

            # predict cluster memberships
            y_pred = predict(X, algorithm)
            ys.append(y_pred)
            del y_pred

            #n_labels0 = len(np.unique(y_pred))
            #X_f, y_pred_f = filter_clustering_noise(X, y_pred)

            # plot
            #plt.subplot(4, len(clustering_algorithms), plot_num)
            #if i_dataset == 0:
            #    plt.title(name, size=18)
            #plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

        y_pred_agglo = agglomerate(X, ys)
        y_pred_all[name] = y_pred_agglo

            # values for metric:
            # From scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]. These metrics support sparse matrix inputs.
            # From scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’] See the documentation for scipy.spatial.distance for details on these metrics. These metrics do not support sparse matrix inputs.

        metrics=['euclidean','cosine','correlation','manhattan','braycurtis','canberra','chebyshev']
        n_labels = len(np.unique(y_pred_agglo))
        for metric in metrics:
            try:
                scores[name][metric] = silhouette_score(X, y_pred_agglo, metric=metric) if n_labels > 1 else float('NaN')
            except:
                print('exception in silhouette score: %s' % sys.exc_info()[0])
                scores[name][metric] = float('nan')

        #    if hasattr(algorithm, 'cluster_centers_'):
        #        centers = algorithm.cluster_centers_
        #        center_colors = colors[:len(centers)]
        #        #plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
        #        y_centers[name] = (centers, center_colors)

            #plt.xlim(-2, 2)
            #plt.ylim(-2, 2)
            #plt.xticks(())
            #plt.yticks(())
            #siltxt=','.join('{:.2f}'.format(x) for x in sil)
            #txt = 'labels={:d}({:d})\nsil={}\n{:.2f}s'.format(n_labels,n_labels0,siltxt,(t1 - t0))
            #plt.text(.99, .01, txt, transform=plt.gca().transAxes, size=15, horizontalalignment='right')
            #plot_num += 1

    # print rankings:
    for name in scores:
        line = name.ljust(35)
        for metric in scores[name]:
            v = '%.3f' % scores[name][metric]
            line += v.rjust(8)
        print(line)

    # aggregate rankings:
    scores_agg = {}
    for name in scores:
        scores_agg[name] = sum(scores[name].values())

    # pick best
    #best_name = max(scores_agg.items(), key=operator.itemgetter(1))[0]
    #print('best = %s' % best_name)

    # pick top-K
    K=6
    final_rank = sorted((x for x in scores_agg.items() if not math.isnan(x[1])), key=operator.itemgetter(1), reverse=True)
    print('final_rank = %s' % final_rank)

    for i in range(K):
        best_name = final_rank[i][0]
        plt.subplot(4, K, K*(plot_num-1)+1+i)
        plt.title(best_name, size=12)
        y_pred = y_pred_all[best_name]
        plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)
        if best_name in y_centers:
            centers, center_colors = y_centers[best_name]
            plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)

    plot_num += 1

plt.show()


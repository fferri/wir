#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import DBSCAN, SpectralClustering

eps = 0.1
gamma1 = 0.01

def cluster(X):
    global X1
    if False:
        X1 = pca(X, gamma1)
    else:
        X1 = X
    if True:
        spectral = SpectralClustering(n_clusters=2, eigen_solver='arpack', gamma=gamma1, affinity='nearest_neighbors').fit(X1)
        labels = spectral.labels_
    else:
        db = DBSCAN(eps=eps, min_samples=20).fit(X1)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        labels[~core_samples_mask] = -1
    return labels

def pca(X, gamma1):
    kpca = KernelPCA(kernel='rbf', fit_inverse_transform=False, gamma=gamma1)
    X_kpca = kpca.fit_transform(X)
    print('X', X.shape)
    print('alphas', kpca.alphas_.shape)
    print('lambdas', kpca.lambdas_.shape)
    #X_back = kpca.inverse_transform(X_kpca)
    return X_kpca

use_synthetic_dataset=True
test_animate=True

if use_synthetic_dataset:
    import dataset.generator
    X, Ytrue = dataset.generator.generate_preset(0)
else:
    import dataset.loader
    X = dataset.loader.load('dataset.csv')


fig = plt.figure()

lines = None

if test_animate:
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    def update(i):
        global X, Y, unique_labels, eps, gamma1, lines, colors
        #eps+=0.0025
        gamma1+=0.1
        Y = cluster(X)
        if lines:
            for k in unique_labels:
                lines[k].set_data([],[])
        unique_labels = set(Y)-set([-1])
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        lines = {k: ax.plot([],[],'.',color=col,markersize=2 if k==-1 else 10)[0] for k, col in tuple(zip(unique_labels, colors))+((-1,'k'),)}
        for k in unique_labels|set([-1]):
            xy = X[Y == k]
            lines[k].set_data(xy[:,0],xy[:,1])
        plt.title('eps={:f}, gamma1={:f}, labels={}'.format(eps, gamma1, sorted(list(unique_labels))))
    anim = animation.FuncAnimation(fig, update, frames=range(100), interval=100, blit=False)
else:
    Y = cluster(X)
    unique_labels = set(Y)
    n_clusters = len(unique_labels-set([-1]))
    print('Number of clusters estimated: %d' % n_clusters)
    print('Labels: %s' % unique_labels)
    unique_labels = set(Y)-set([-1])
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k in unique_labels|set([-1]):
        xy=X[Y==k]
        plt.scatter(xy[:,0],xy[:,1],color='k' if k==-1 else colors[k])


plt.show()


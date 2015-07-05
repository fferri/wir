#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import DBSCAN

def cluster(X):
    db = DBSCAN(eps=0.6, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    labels[~core_samples_mask] = -1
    return labels

def pca(X, gamma1):
    kpca = KernelPCA(kernel='rbf', fit_inverse_transform=False, gamma=10*gamma1)
    X_kpca = kpca.fit_transform(X)
    #X_back = kpca.inverse_transform(X_kpca)
    return X_kpca

use_synthetic_dataset=False
test_animate=True

if use_synthetic_dataset:
    import dataset.generator
    X, Y = dataset.generator.generate_preset(0)
else:
    import dataset.loader
    X = dataset.loader.load('dataset.csv')
    Y = cluster(X)

unique_labels = set(Y)
n_clusters = len(unique_labels-set([-1]))
print('Number of clusters estimated: %d' % n_clusters)
print('Labels: %s' % unique_labels)

fig = plt.figure()

colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

if test_animate:
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    lines = {k: ax.plot([],[],'.',markerfacecolor=(col if k != -1 else 'k'), markersize=(10 if k != -1 else 3))[0] for k, col in zip(unique_labels, colors)}
    def init():
        pass
    def update(i):
        gamma1=i*0.01
        xy_all = pca(X, gamma1)
        for k in unique_labels:
            xy = xy_all[Y == k]
            lines[k].set_data(xy[:,0],xy[:,1])
        plt.title('gamma=%f' % gamma1)
    anim = animation.FuncAnimation(fig, update, frames=range(100), init_func=init, interval=100, blit=False)
else:
    X1=np.array(X)
    Y1=np.array(Y)
    for k in unique_labels:
        xy=X1[Y1==k]
        plt.scatter(xy[:,0],xy[:,1],color=colors[k])


plt.show()


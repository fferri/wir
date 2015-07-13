import numpy as np

class AgglomerativeClustering:
    '''
    Compute Agglomerative Clustering

    Usage:
        ac = AgglomerativeClustering(X)
        ac.add_clustering(Y1)
        ac.add_clustering(Y2)
        ...
        Y = ac.compute()
    '''

    def __init__(self, X):
        self.X = X
        self.n = X.shape[0]
        self.Ys = {}

    def add_clustering(self, Y, name=None):
        if name is None: name = len(self.Ys)
        if Y.shape[0] != self.n:
            raise Exception('bad number of rows: {}. should be {}'.format(Y.shape[0], self.n))
        self.Ys[name] = Y

    def compute_weight(self, i, j):
        return sum(Y[i] != Y[j] for Y in self.Ys.values()) / len(self.Ys)

    def compute_weights(self):
        weights = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                w = self.compute_weight(i, j)
                weights[i,j] = w
                weights[j,i] = w
        return weights

    def compute_average_distance(self, Y, cluster_i, cluster_j):
        dist = 0
        num = 0
        for i in [idx for idx, y in enumerate(Y) if y == cluster_i]:
            for j in [idx for idx, y in enumerate(Y) if y == cluster_j]:
                dist += self.weights[i,j]
                num += 1
        dist /= num
        return dist

    def compute_average_distances(self, Y):
        clusters = list(set(Y))
        l = len(clusters)
        d = []
        for i in range(l):
            for j in range(i, l):
                if i == j: continue
                ci, cj = clusters[i], clusters[j]
                d.append((self.compute_average_distance(Y, ci, cj), ci, cj))
        return d

    def merge_clusters(self, Y, cluster_i, cluster_j):
        if cluster_i == cluster_j:
            raise Exception('merge_clusters() called on the same cluster {}'.format(cluster_i))
        new_cluster = 1 + max(Y)
        return np.array([new_cluster if y in (cluster_i, cluster_j) else y for y in Y]).reshape(Y.shape)

    def compute(self):
        self.weights = self.compute_weights()
        Y = self.n + 1000 + np.arange(self.n)
        while True:
            m = min(self.compute_average_distances(Y))
            dist, ci, cj = m
            if dist < 0.5:
                Y = self.merge_clusters(Y, ci, cj)
            else:
                break
        # use cluster numbers from 0 to k-1
        mapping={v:k for k,v in enumerate(set(Y))}
        return np.array([mapping[y] for y in Y])


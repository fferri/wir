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
        #if Y.shape[0] != self.n:
            #raise Exception('bad number of rows: {}. should be {}'.format(Y.shape[0], self.n))
        self.Ys[name] = Y

    def compute_weight(self, i, j):
        if i == j: return 1.0
        return sum(Y[i] != Y[j] for Y in self.Ys.values()) / len(self.Ys)

    def compute_weights(self):
        weights = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                w = self.compute_weight(i, j)
                weights[i,j] = w
                weights[j,i] = w
        return weights

    def compute(self):
        from random import sample

        n = len(self.X)
        w = self.compute_weights()
        V = set(range(n))
        C = [-1] * n
        nc = 0

        while V:
            print('V=%s' % V)
            v = sample(V, 1)[0]
            used_cluster = False
            for i in range(n):
                if C[i] == -1 and w[i,v] >= 0.5 and i in V:
                    C[i] = nc
                    used_cluster = True
                    V.remove(i)
            if used_cluster: nc += 1

        return C

if __name__ == '__main__':
    X  = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    C1 = np.array([ 1,  1,  1,  1,  2,  2,  2,  2])
    C2 = np.array([ 1,  1,  1,  2,  2,  2,  3,  3])
    C3 = np.array([ 1,  2,  1,  1,  2,  3,  3,  3])
    ac = AgglomerativeClustering(X)
    for c in (C1, C2, C3):
        ac.add_clustering(c)
    print(ac.compute())

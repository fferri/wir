import numpy as np

class RankAggregation:
    '''
    Compute Rank Aggregation

    Usage:
        ra = RankAggregation(X)
        ra.add_ranker(R1)
        ra.add_ranker(R2)
        # e.g.:
        ra.add_ranker(lambda x, y: silhouette_score(x, y, metric='euclidean'))
        ...
        Z1, Z2, ... = ra.compute(Y1, Y2, ...)
    '''

    def __init__(self, X):
        self.X = X
        if X: self.n = X.shape[0]
        self.Ys = {}
        self.rankers = {}

    def add_ranker(self, r, name=None):
        if name is None: name = len(self.rankers)
        self.rankers[name] = r

    def compute_rank(self, ranker, *args):
        v = [ranker(self.X, arg) for arg in args]
        n = len(v)
        d = dict(zip(sorted(v), range(n)))
        return [1+d[i] for i in v]

    def compute(self, *args):
        ranks = []
        for rn, r in self.rankers.items():
            rank = self.compute_rank(r, *args)
            ranks.append(rank)

        result = []
        for i in range(len(args)):
            result.append((-sum(ranks[j][i] for j in range(len(ranks))), args[i]))
        result.sort()
        return [x[1] for x in result]

if __name__ == '__main__':
    v = [0.1, 4.2, 0.3, 3.1, 1.2]
    ra = RankAggregation(None)
    ra.add_ranker(lambda x, y: y, 'r1')
    ra.add_ranker(lambda x, y: y*y-0.2*y, 'r2')
    print(ra.compute(*v))

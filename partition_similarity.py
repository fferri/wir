from functools import reduce
from munkres import Munkres, print_matrix

def is_partitioning(p, valid_sequence_type=(list, tuple), verbose=False):
    if not isinstance(p, valid_sequence_type):
        if verbose: print('is_partitioning: not a partition because not a sequence: {}'.format(p))
        return False
    if not all(isinstance(q, valid_sequence_type) for q in p):
        if verbose: print('is_partitioning: not a partition because some item is not a sequence: {}'.format(p))
        return False
    if not all(p):
        if verbose: print('is_partitioning: not a partition because it contains an empty sequence: {}'.format(p))
        return False
    f = sorted([r for q in p for r in q])
    sf = sorted(list(set(f)))
    if verbose: print('is_partitioning: f={}, sf={}'.format(f, sf))
    return f == sf

def jaccard_index(a, b):
    a, b = set(a), set(b)
    j = len(a & b) / len(a | b)
    #print('jaccard_index({}, {}) = {}'.format(a, b, j))
    return j

def match_partitions(p1, p2):
    if not is_partitioning(p1):
        raise Exception('not a partitioning: {}'.format(p1))
    if not is_partitioning(p2):
        raise Exception('not a partitioning: {}'.format(p2))
    n = len(p1)
    m = len(p2)
    c = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            c[i][j] = 1 - jaccard_index(p1[i], p2[j])
    mk = Munkres()
    indices = mk.compute(c)
    total = 0
    for i, j in indices:
        v = 1 - c[i][j]
        try:
            print('{} --> {}   ({})'.format(p1[i], p2[j], v))
        except IndexError:
            print('# {} --> {}    (0.0)'.format(i, j))
        total += v
    total /= len(indices)
    print('total: {}'.format(total))


inf = float('inf')
sc = [
    [0.346, 0.545, 0.612, 0.420, 0.634, 0.585, 0.595],
    [0.346, 0.759, 0.788, 0.626, 0.797, 0.775, 0.938],
    [0.048, 0.899, 0.794, 0.709, 0.757, 0.823, 0.913],
    [0.346, 0.759, 0.788, 0.626, 0.797, 0.775, 0.938],
    [0.346, 0.759, 0.788, 0.626, 0.797, 0.775, 0.938],
    [inf, 0.641, 0.691, 0.536, 0.696, 0.683, 0.867],
    [inf, 0.581, 0.641, 0.453, 0.645, 0.633, 0.674],
    [0.346, 0.759, 0.788, 0.626, 0.797, 0.775, 0.938],
    [0.065, -0.515, -0.184, 0.073, -0.101, -0.239, -0.229],
    [0.271, 0.649, 0.693, 0.539, 0.698, 0.682, 0.859]
]
def col(i):
    return list((sc[j][i],j) for j in range(len(sc)))
def rank(i):
    c = sorted(col(i))
    r = [0] * len(c)
    for i,x in enumerate(c):
        r[x[1]] = i
    return r
ranks = [rank(i) for i in range(len(sc[0]))]
finalrank = [sum(ranks[i][j] for i in range(len(ranks))) for j in range(len(ranks[0]))]
print(finalrank)

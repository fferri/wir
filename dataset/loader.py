import csv
from sklearn.preprocessing import StandardScaler

def load(filename):
    X = None
    with open(filename, 'r') as csvfile:
        r = csv.reader(csvfile)
        for row in r:
            if X is None:
                X = []
                continue
            X.append(list(map(float, row[1:])))
    X = StandardScaler().fit_transform(X)
    return X

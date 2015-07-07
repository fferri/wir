import random, math
import numpy as np

def generate(gen_type='radial', size=(150,250), radius=((0.5,0.5), (1.0,1.0)), arange=((0,3),(2,6)), center=((0,0),(0,0)), noise=((0.1,0.5),(0.1,0.5))):
    k = len(size)
    ls = set(len(x) for x in (radius, arange, center, noise))
    if ls != set([k]):
        raise Exception('size, radius and arange must have equal length')
    if set(len(x) for x in radius+arange+center+noise) != set([2]):
        raise Exception('radius[i] and arange[i] must have length == 2 for all i')
    X = []
    Y = []
    ks = list(range(k))
    ns = [0]*k
    n = sum(size)
    def frandrange(range1):
        m, M = min(range1), max(range1)
        delta = M - m
        return random.random() * delta + m
    while len(X) < n:
        ki = random.choice(ks)
        if ns[ki] < size[ki]:
            r = frandrange(radius[ki]) + np.random.normal(0,noise[ki][0])
            a = frandrange(arange[ki]) + np.random.normal(0,noise[ki][1])
            x = center[ki][0] + r * math.cos(a)
            y = center[ki][1] + r * math.sin(a)
            X.append([x,y])
            Y.append(ki)
            ns[ki] += 1
        else:
            del ks[ki]
    return np.array(X), np.array(Y)

preset = [
        {
            'gen_type':  'radial',
            'size':      (           600,             900),
            'radius':    (( 0.50,  0.50),  ( 1.00,  1.00)),
            'arange':    (( 0.00,  3.10),  ( 2.00,  6.28)),
            'center':    (( 0.00,  0.00),  ( 0.00,  0.00)),
            'noise':     (( 0.08,  0.50),  ( 0.10,  0.20))
        },
        {
            'gen_type':  'radial',
            'size':      (          600,            900),
            'radius':    (( 0.50, 0.50),  ( 0.50, 0.50)),
            'arange':    ((-0.60, 3.80),  ( 2.40, 6.60)),
            'center':    ((-0.10, 0.00),  ( 0.35,-0.30)),
            'noise':     (( 0.08, 0.20),  ( 0.09, 0.20))
        }
    ]

def generate_preset(i):
    return generate(**preset[i])

import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Both are parameters to be fitted
N = 1000
C = 0.001

def main():
    start = numpy.zeros(N)
    ret = minimize(objective ,start,bounds=[(0,C) for b in range(N)], constraints={'type':'eq', 'fun':zerofun})
    #alpha = ret[’x’]


def objective():
    pass

def zerofun():
    pass

def indicator():
    pass

if __name__ == '__main__':
    main()
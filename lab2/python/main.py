import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import statistics

# Both are parameters to be fitted
N = 1000
C = 0.001

def main():
    start = numpy.zeros(N)
    ret = minimize(objective ,start,bounds=[(0,C) for b in range(N)], constraints={'type':'eq', 'fun':zerofun})
    #alpha = ret[’x’]

def linear_kernel(x, y):
    """ Returns the scalar product. """
    return np.dot(x, y.T)

def polynomial_kernel(x, y, p=3):
    """ Returns (x.y + 1)^p """
    return (np.dot(x, y.T) + 1) ** p

def rbf_kernel(x, y, sigma=0.5):
    """ Returns exp(-||x-y||^2 / (2*sigma^2)) """
    # Efficient pairwise squared distance
    x_sq = np.sum(x**2, axis=1).reshape(-1, 1)
    y_sq = np.sum(y**2, axis=1).reshape(1, -1)
    dist_sq = x_sq + y_sq - 2 * np.dot(x, y.T)
    # Clip small negatives due to precision
    dist_sq = np.maximum(dist_sq, 0)
    return np.exp(-dist_sq / (2 * sigma**2))

def sigma(inputs):
    distance = []
    for i in range(len(inputs)):
        for j in range(len(inputs)):
            distance.append(numpy.linalg.norm(inputs[i]-inputs[j]))
    return statistics.median(distance)
    


def objective():
    pass

def zerofun():
    pass

def indicator():
    pass

if __name__ == '__main__':
    main()
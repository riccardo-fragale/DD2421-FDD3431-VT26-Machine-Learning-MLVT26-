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

def linear_kernel(x):
    # the faster way to do the Xt * y where y=x is to do x * xT
    return numpy.dot(x,x.T)

def polynomial_kernel(x,p):
    return math.pow(numpy.dot(x,x.T)+1,p)

def sigma(inputs):
    distance = []
    for i in range(len(inputs)):
        for j in range(len(inputs)):
            distance.append(numpy.linalg.norm(inputs[i]-inputs[j]))
    return statistics.median(distance)
    
def radial_basis_function(x,y,sigma):
    return math.exp(-numpy.linalg.norm(x-y)/(2*math.pow(sigma,2)))


def objective():
    pass

def zerofun():
    pass

def indicator():
    pass

if __name__ == '__main__':
    main()
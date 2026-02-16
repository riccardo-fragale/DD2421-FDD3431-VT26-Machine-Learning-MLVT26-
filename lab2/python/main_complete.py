import numpy as np
import matplotlib.pyplot as plt
import random
import statistics
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist

def generate_data():
    np.random.seed(100) # same seed to make tests reproducibles
    
    # Class creation
    classA = np.concatenate((
        np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
        np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]
    ))
    
    classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
    
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((
        np.ones(classA.shape[0]),
        -np.ones(classB.shape[0])
    ))
    
    # shuffling the data
    N = inputs.shape[0]
    permute = list(range(N))
    random.seed(100)
    random.shuffle(permute)
    
    # Creating inputs and targets
    inputs = inputs[permute, :]
    targets = targets[permute]
    
    return inputs, targets

# linear kernel
def linear_kernel(x, y):
    return np.dot(x, y.T)

# polynomial kernel
def polynomial_kernel(x, y, p=3):
    return (np.dot(x, y.T) + 1) ** p

# rbf kernel
def rbf_kernel(x, y, sigma=0.5):
    # Compute pairwise euclidean distance 
    dists = cdist(x, y, 'sqeuclidean')
    return np.exp(-dists / (2 * sigma**2))

# split kernel creation based on kernel type
def get_kernel_matrix(X, kernel_type, **kwargs):
    if kernel_type == 'linear':
        return linear_kernel(X, X)
    elif kernel_type == 'polynomial':
        return polynomial_kernel(X, X, p=kwargs.get('p', 3))
    elif kernel_type == 'rbf':
        # Sigma is handled in train_svm, but safety value 0.5
        return rbf_kernel(X, X, sigma=kwargs.get('sigma', 0.5))
    else:
        raise ValueError("Unknown kernel type")

# Training
def train_svm(inputs, targets, C, kernel_type='linear', **kwargs):   #kernel type is by default linear but could be changed
    # sigma heuristic for RBF
    if kernel_type == 'rbf' and kwargs.get('sigma') is None:
        # compute all pairwise euclidean distances
        distances = pdist(inputs, 'euclidean')
        # sigma = median distance
        sigma = statistics.median(distances)
        kwargs['sigma'] = sigma
        print(f"Calculated sigma = {sigma:.4f}")
    
    N = inputs.shape[0]
    K = get_kernel_matrix(inputs, kernel_type, **kwargs)
    P = np.outer(targets, targets) * K
    
    # dual objective function
    def objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)
    
    # respecting the contraint sum(alpha * targets) = 0
    def zerofun(alpha):
        return np.dot(alpha, targets)
    
    bounds = [(0, C) for _ in range(N)]
    constraints = {'type': 'eq', 'fun': zerofun}
    start = np.zeros(N)
    
    # minimizing
    ret = minimize(objective, start, bounds=bounds, constraints=constraints)
    if not ret.success:
        print("Optimization did not converge.")
        
    alpha = ret.x
    
    # extract the support vectors
    sv_indices = alpha > 1e-5
    support_vectors = inputs[sv_indices]
    support_alpha = alpha[sv_indices]
    support_targets = targets[sv_indices]
    
    # average over margin support vectors (0 < alpha < C)
    margin_indices = (alpha > 1e-5) & (alpha < C - 1e-5)
    
    if np.sum(margin_indices) > 0:
        b_values = []
        for i in np.where(margin_indices)[0]:
            prediction_no_b = np.dot(alpha * targets, K[:, i])
            b_values.append(targets[i] - prediction_no_b)
        b = np.mean(b_values)
    else:
        b = 0.0 

    # return tuple containing all the info needed 
    return {
        'alpha': alpha,
        'b': b,
        'sv': support_vectors,
        'sv_alpha': support_alpha,
        'sv_targets': support_targets,
        'kernel_type': kernel_type,
        'kwargs': kwargs 
    }

# Indicator function
def indicator(model, x_new):
    sv = model['sv']
    sv_alpha = model['sv_alpha']
    sv_targets = model['sv_targets']
    b = model['b']
    kernel_type = model['kernel_type']
    kwargs = model['kwargs']
    
    if kernel_type == 'linear':
        k_val = linear_kernel(sv, x_new)
    elif kernel_type == 'polynomial':
        k_val = polynomial_kernel(sv, x_new, p=kwargs.get('p', 3))
    elif kernel_type == 'rbf':
        k_val = rbf_kernel(sv, x_new, sigma=kwargs.get('sigma', 0.5))
        
    weights = sv_alpha * sv_targets
    return np.dot(weights, k_val) + b

# plotting
def plot_results_enhanced(inputs, targets, model):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # background and grid
    ax.set_facecolor('#f0f0f5') 
    ax.grid(True, linestyle='--', color='white', linewidth=1.5, alpha=0.8)
    
    # data points
    ax.plot([p[0] for p in inputs[targets==1]], 
            [p[1] for p in inputs[targets==1]], 
            'b.', markersize=12, label='Class A (+1)')
    
    ax.plot([p[0] for p in inputs[targets==-1]], 
            [p[1] for p in inputs[targets==-1]], 
            'r.', markersize=12, label='Class B (-1)')
    
    # highlight support vectors
    ax.scatter(model['sv'][:, 0], model['sv'][:, 1], 
               s=200, facecolors='none', edgecolors='g', linewidth=2, 
               label='Support Vectors')
    
    # decision boundary and margins
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -1.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = indicator(model, grid_points)
    Z = Z.reshape(xx.shape)
    
    CS = ax.contour(xx, yy, Z, levels=[-1, 0, 1], 
                    colors=['k', 'k', 'k'], 
                    linestyles=['--', '-', '--'],
                    linewidths=[1.5, 2.5, 1.5])
    
    ax.clabel(CS, inline=True, fontsize=10, fmt={-1:'-1', 0:'0', 1:'+1'})
    
    font_title = {'family': 'serif', 'color':  'darkred', 'weight': 'bold', 'size': 16}
    font_label = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 14}
    
    ax.set_title(f"SVM Decision Boundary ({model['kernel_type']} kernel)", fontdict=font_title)
    ax.set_xlabel("Feature $x_1$", fontdict=font_label)
    ax.set_ylabel("Feature $x_2$", fontdict=font_label)
    
    ax.axis('equal') 
    
    legend = ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.setp(legend.get_texts(), family='serif')
    
    plt.savefig('svmplot_enhanced.pdf') 
    plt.show() 


if __name__ == "__main__":
    inputs, targets = generate_data()
    
    # kernel type modifiable
    model = train_svm(inputs, targets, C=10.0, kernel_type='polynomial') 
    
    plot_results_enhanced(inputs, targets, model)
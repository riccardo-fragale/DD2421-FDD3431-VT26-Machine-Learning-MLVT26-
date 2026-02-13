import numpy as np
import matplotlib.pyplot as plt
import random
import statistics
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist

# --- 1. Data Generation ---
def generate_data():
    np.random.seed(100)
    classA = np.concatenate((
        np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
        np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]
    ))
    classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
    N = inputs.shape[0]
    permute = list(range(N))
    random.seed(100)
    random.shuffle(permute)
    return inputs[permute, :], targets[permute]

# --- 2. Kernel Functions ---
def linear_kernel(x, y):
    return np.dot(x, y.T)

def polynomial_kernel(x, y, p=3):
    return (np.dot(x, y.T) + 1) ** p

def rbf_kernel(x, y, sigma=0.5):
    dists = cdist(x, y, 'sqeuclidean')
    return np.exp(-dists / (2 * sigma**2))

def get_kernel_matrix(X, kernel_type, **kwargs):
    if kernel_type == 'linear': return linear_kernel(X, X)
    elif kernel_type == 'polynomial': return polynomial_kernel(X, X, p=kwargs.get('p', 3))
    elif kernel_type == 'rbf': return rbf_kernel(X, X, sigma=kwargs.get('sigma', 0.5))
    else: raise ValueError("Unknown kernel")

# --- 3. SVM Training ---
def train_svm(inputs, targets, C, kernel_type='linear', **kwargs):
    if kernel_type == 'rbf' and kwargs.get('sigma') is None:
        distances = pdist(inputs, 'euclidean')
        kwargs['sigma'] = statistics.median(distances)

    N = inputs.shape[0]
    K = get_kernel_matrix(inputs, kernel_type, **kwargs)
    P = np.outer(targets, targets) * K
    
    def objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)
    def zerofun(alpha):
        return np.dot(alpha, targets)
    
    bounds = [(0, C) for _ in range(N)]
    constraints = {'type': 'eq', 'fun': zerofun}
    
    ret = minimize(objective, np.zeros(N), bounds=bounds, constraints=constraints)
    alpha = ret.x
    
    sv_indices = alpha > 1e-5
    sv = inputs[sv_indices]
    sv_alpha = alpha[sv_indices]
    sv_targets = targets[sv_indices]
    
    margin_indices = (alpha > 1e-5) & (alpha < C - 1e-5)
    if np.sum(margin_indices) > 0:
        idx = np.where(margin_indices)[0]
        b = np.mean([targets[i] - np.dot(alpha * targets, K[:, i]) for i in idx])
    else:
        b = 0.0
        
    return {'sv': sv, 'sv_alpha': sv_alpha, 'sv_targets': sv_targets, 
            'b': b, 'kernel_type': kernel_type, 'kwargs': kwargs}

# --- 4. Indicator Function ---
def indicator(model, x_new):
    sv, sv_a, sv_t = model['sv'], model['sv_alpha'], model['sv_targets']
    k_type, kwargs = model['kernel_type'], model['kwargs']
    if k_type == 'linear': k_val = linear_kernel(sv, x_new)
    elif k_type == 'polynomial': k_val = polynomial_kernel(sv, x_new, p=kwargs.get('p', 3))
    elif k_type == 'rbf': k_val = rbf_kernel(sv, x_new, sigma=kwargs.get('sigma', 0.5))
    return np.dot(sv_a * sv_t, k_val) + model['b']

# --- 5. Plotting ---
def plot_subplot(ax, inputs, targets, model, title):
    ax.set_facecolor('#f8f8f8')
    ax.plot(inputs[targets==1, 0], inputs[targets==1, 1], 'b.', markersize=6)
    ax.plot(inputs[targets==-1, 0], inputs[targets==-1, 1], 'r.', markersize=6)
    ax.scatter(model['sv'][:, 0], model['sv'][:, 1], s=50, facecolors='none', edgecolors='g', alpha=0.6)
    
    x_min, x_max, y_min, y_max = -2.5, 2.5, -1.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = indicator(model, np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors='k', linestyles=['--', '-', '--'], linewidths=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

# --- Main Execution ---
if __name__ == "__main__":
    inputs, targets = generate_data()
    C_values = [0.1, 1, 100]
    kernels = ['linear', 'polynomial', 'rbf']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    
    for row_idx, k_type in enumerate(kernels):
        for col_idx, C in enumerate(C_values):
            print(f"Training {k_type} with C={C}...")
            # Use default p=3 for poly and heuristic sigma for RBF
            model = train_svm(inputs, targets, C=C, kernel_type=k_type)
            plot_subplot(axes[row_idx, col_idx], inputs, targets, model, f"{k_type.upper()} | C={C}")
    
    plt.savefig('svm_full_comparison.pdf')
    plt.show()
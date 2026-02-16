import numpy as np
import matplotlib.pyplot as plt
import random
import statistics
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist

# flexible data generation 
def generate_data(
    classA_centers=[(1.5, 0.5), (-1.5, 0.5)], 
    classB_center=(0.0, -0.5), 
    spread=0.2, 
    seed=100
):
    """
    Generates data with custom cluster positions and spread (std deviation).
    classA_centers: List of (x, y) tuples for Class A clusters.
    classB_center: (x, y) tuple for Class B cluster.
    spread: Standard deviation (radius) of the clusters.
    """
    np.random.seed(seed) 
    
    # generate class A (combines multiple clusters)
    classA_parts = []
    for center in classA_centers:
        # Generate 10 points per center for A
        part = np.random.randn(10, 2) * spread + center
        classA_parts.append(part)
    classA = np.concatenate(classA_parts)
    
    # generate class B (single cluster of 20 points)
    classB = np.random.randn(20, 2) * spread + classB_center
    
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((
        np.ones(classA.shape[0]),
        -np.ones(classB.shape[0])
    ))
    
    # shuffling
    N = inputs.shape[0]
    permute = list(range(N))
    random.seed(seed)
    random.shuffle(permute)
    
    inputs = inputs[permute, :]
    targets = targets[permute]
    
    return inputs, targets

# kernels
def linear_kernel(x, y):
    return np.dot(x, y.T)

def polynomial_kernel(x, y, p=3):
    return (np.dot(x, y.T) + 1) ** p

def rbf_kernel(x, y, sigma=0.5):
    dists = cdist(x, y, 'sqeuclidean')
    return np.exp(-dists / (2 * sigma**2))

def get_kernel_matrix(X, kernel_type, **kwargs):
    if kernel_type == 'linear':
        return linear_kernel(X, X)
    elif kernel_type == 'polynomial':
        return polynomial_kernel(X, X, p=kwargs.get('p', 3))
    elif kernel_type == 'rbf':
        return rbf_kernel(X, X, sigma=kwargs.get('sigma', 0.5))
    else:
        raise ValueError("Unknown kernel type")


def train_svm(inputs, targets, C, kernel_type='linear', **kwargs):
    # heuristic for sigma if using RBF
    if kernel_type == 'rbf' and kwargs.get('sigma') is None:
        distances = pdist(inputs, 'euclidean')
        sigma = statistics.median(distances)
        kwargs['sigma'] = sigma
        print(f"   [Heuristic] Calculated sigma: {sigma:.4f}")

    N = inputs.shape[0]
    K = get_kernel_matrix(inputs, kernel_type, **kwargs)
    P = np.outer(targets, targets) * K
    
    # dual objective
    def objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)
    
    def zerofun(alpha):
        return np.dot(alpha, targets)
    
    bounds = [(0, C) for _ in range(N)]
    constraints = {'type': 'eq', 'fun': zerofun}
    start = np.zeros(N)
    
    print(f"   Optimizing with {kernel_type} kernel...")
    ret = minimize(objective, start, bounds=bounds, constraints=constraints)
    
    if not ret.success:
        print(f"   WARNING: Optimization FAILED! Message: {ret.message}")
    else:
        print(f"   Optimization success. Iterations: {ret.nit}")

    alpha = ret.x
    
    # support vectors
    sv_indices = alpha > 1e-5
    support_vectors = inputs[sv_indices]
    support_alpha = alpha[sv_indices]
    support_targets = targets[sv_indices]
    
    # bias calculation
    margin_indices = (alpha > 1e-5) & (alpha < C - 1e-5)
    if np.sum(margin_indices) > 0:
        b_values = []
        for i in np.where(margin_indices)[0]:
            prediction_no_b = np.dot(alpha * targets, K[:, i])
            b_values.append(targets[i] - prediction_no_b)
        b = np.mean(b_values)
    else:
        b = 0.0 
        print("   Note: No free support vectors found (Hard Separation or Overfitting). b set to 0.0")

    return {
        'alpha': alpha, 'b': b, 'sv': support_vectors,
        'sv_alpha': support_alpha, 'sv_targets': support_targets,
        'kernel_type': kernel_type, 'kwargs': kwargs 
    }

# indicator function
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
def plot_results_enhanced(inputs, targets, model, title_suffix, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('#f0f0f5') 
    ax.grid(True, linestyle='--', color='white', linewidth=1.5, alpha=0.8)
    
    
    ax.plot([p[0] for p in inputs[targets==1]], [p[1] for p in inputs[targets==1]], 
            'b.', markersize=12, label='Class A (+1)')
    ax.plot([p[0] for p in inputs[targets==-1]], [p[1] for p in inputs[targets==-1]], 
            'r.', markersize=12, label='Class B (-1)')
    
    
    if len(model['sv']) > 0:
        ax.scatter(model['sv'][:, 0], model['sv'][:, 1], 
                   s=150, facecolors='none', edgecolors='g', linewidth=2, label='Support Vectors')
    
    
    x_min, x_max = np.min(inputs[:,0])-1, np.max(inputs[:,0])+1
    y_min, y_max = np.min(inputs[:,1])-1, np.max(inputs[:,1])+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = indicator(model, grid_points)
    Z = Z.reshape(xx.shape)
    
    CS = ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['k', 'k', 'k'], 
                    linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
    ax.clabel(CS, inline=True, fontsize=10)
    
    ax.set_title(f"SVM: {title_suffix} ({model['kernel_type']})", fontsize=14, weight='bold')
    ax.axis('equal')
    ax.legend(facecolor='white', framealpha=0.9)
    
    print(f"   Saving plot to {filename}...")
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    
    
    print("\n RUN 1: Easy Configuration (Linearly Separable) ")
    inputs_1, targets_1 = generate_data(
        classA_centers=[(2.0, 2.0), (-2.0, 2.0)], # Moved up and out
        classB_center=(0.0, -2.0),                # Moved down
        spread=0.1                                # Tighter clusters
    )
    # using linear kernel
    model_1 = train_svm(inputs_1, targets_1, C=100.0, kernel_type='linear')
    plot_results_enhanced(inputs_1, targets_1, model_1, "Easy Separation", "svm_experiment_1.pdf")

    # experiment 2: medium/hard (closer, larger spread) ---
    print("\n RUN 2: Hard Configuration (closer, larger spread) ")
    inputs_2, targets_2 = generate_data(
        classA_centers=[(1.0, 0.5), (-1.0, 0.5)], # Closer to center
        classB_center=(0.0, 0.0),                 # Right in the middle
        spread=0.4                                # Messier
    )
    # RBF Kernel (linear would fail here)
    model_2 = train_svm(inputs_2, targets_2, C=10.0, kernel_type='rbf')
    plot_results_enhanced(inputs_2, targets_2, model_2, "Overlapping Clusters", "svm_experiment_2.pdf")

    # Experiment 3: very hard (high overlap - testing limits)
    print("\n RUN 3: very hard (high overlap) ")
    inputs_3, targets_3 = generate_data(
        classA_centers=[(0.5, 0.5), (-0.5, 0.5)], # Very tight
        classB_center=(0.0, 0.2),                 # Almost on top of A
        spread=0.6                                # Huge spread, lots of noise
    )
    # RBF with high C to try and force fit, or low C to ignore errors
    model_3 = train_svm(inputs_3, targets_3, C=1.0, kernel_type='rbf')
    plot_results_enhanced(inputs_3, targets_3, model_3, "High Noise/Overlap", "svm_experiment_3.pdf")
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
    # Heuristic for RBF if sigma is not provided
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
    support_vectors = inputs[sv_indices]
    support_alpha = alpha[sv_indices]
    support_targets = targets[sv_indices]
    
    # Bias calculation
    margin_indices = (alpha > 1e-5) & (alpha < C - 1e-5)
    if np.sum(margin_indices) > 0:
        b_values = []
        for i in np.where(margin_indices)[0]:
            pred = np.dot(alpha * targets, K[:, i])
            b_values.append(targets[i] - pred)
        b = np.mean(b_values)
    else:
        b = 0.0
        
    return { 'alpha': alpha, 'b': b, 'sv': support_vectors, 
             'sv_alpha': support_alpha, 'sv_targets': support_targets, 
             'kernel_type': kernel_type, 'kwargs': kwargs }

# --- 4. Indicator Function ---
def indicator(model, x_new):
    sv, sv_alpha, sv_t = model['sv'], model['sv_alpha'], model['sv_targets']
    k_type, kwargs = model['kernel_type'], model['kwargs']
    
    if k_type == 'linear': k_val = linear_kernel(sv, x_new)
    elif k_type == 'polynomial': k_val = polynomial_kernel(sv, x_new, p=kwargs.get('p'))
    elif k_type == 'rbf': k_val = rbf_kernel(sv, x_new, sigma=kwargs.get('sigma'))
        
    return np.dot(sv_alpha * sv_t, k_val) + model['b']

# --- 5. Multi-Plotting Function ---
def plot_subplot(ax, inputs, targets, model, title):
    # Background
    ax.set_facecolor('#f0f0f5')
    
    # Data
    ax.plot([p[0] for p in inputs[targets==1]], [p[1] for p in inputs[targets==1]], 'b.', label='A')
    ax.plot([p[0] for p in inputs[targets==-1]], [p[1] for p in inputs[targets==-1]], 'r.', label='B')
    
    # SVs
    ax.scatter(model['sv'][:, 0], model['sv'][:, 1], s=100, facecolors='none', edgecolors='g', linewidth=1.5)
    
    # Boundary
    x_min, x_max, y_min, y_max = -2.5, 2.5, -1.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = indicator(model, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors='k', linestyles=['--', '-', '--'], linewidths=[0.5, 1.5, 0.5])
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# --- Main Execution ---
if __name__ == "__main__":
    inputs, targets = generate_data()
    
    # Setup 2x3 Grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plt.subplots_adjust(hspace=0.3)
    
    # --- ROW 1: Polynomial Kernel (Varying p) ---
    p_values = [1, 2, 5]
    for i, p in enumerate(p_values):
        print(f"Training Poly p={p}...")
        model = train_svm(inputs, targets, C=10.0, kernel_type='polynomial', p=p)
        plot_subplot(axes[0, i], inputs, targets, model, f"Poly Kernel (p={p})")

    # --- ROW 2: RBF Kernel (Varying Sigma) ---
    sigma_values = [0.2, 1.0, 4.0] # Small, Medium, Large
    for i, sigma in enumerate(sigma_values):
        print(f"Training RBF sigma={sigma}...")
        model = train_svm(inputs, targets, C=10.0, kernel_type='rbf', sigma=sigma)
        plot_subplot(axes[1, i], inputs, targets, model, f"RBF Kernel (σ={sigma})")

    plt.savefig('svm_bias_variance_grid.pdf')
    plt.show()
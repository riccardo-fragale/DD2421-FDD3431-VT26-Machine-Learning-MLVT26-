import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def generate_noisy_data():
    np.random.seed(42)
    # overlapping clusters
    classA = np.random.randn(20, 2) * 0.5 + [0.7, 0.7]
    classB = np.random.randn(20, 2) * 0.5 + [0.0, 0.0]
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(20), -np.ones(20)))
    return inputs, targets

# kernel and svm logic 
def rbf_kernel(x, y, sigma=0.5):
    return np.exp(-cdist(x, y, 'sqeuclidean') / (2 * sigma**2))

def train_svm_simple(inputs, targets, C, kernel_func):
    N = len(targets)
    K = kernel_func(inputs, inputs)
    P = np.outer(targets, targets) * K
    def objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)
    cons = {'type': 'eq', 'fun': lambda a: np.dot(a, targets)}
    res = minimize(objective, np.zeros(N), bounds=[(0, C)]*N, constraints=cons)
    return res.x, K

def plot_comparison(inputs, targets):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # more slack (low C)
    alpha1, _ = train_svm_simple(inputs, targets, 0.01, lambda x, y: np.dot(x, y.T))
    # more complexity (high C, RBF)
    alpha2, _ = train_svm_simple(inputs, targets, 100, lambda x, y: rbf_kernel(x, y, sigma=0.2))

    # plot boundaries
    for ax, alpha, title, k_func in zip([ax1, ax2], [alpha1, alpha2], 
                                       ["Strategy: More Slack (Low C)", "Strategy: High Complexity (RBF)"],
                                       [lambda x, y: np.dot(x, y.T), lambda x, y: rbf_kernel(x, y, sigma=0.2)]):
        ax.scatter(inputs[:,0], inputs[:,1], c=targets, cmap='bwr', edgecolors='k')
        x_min, x_max = inputs[:,0].min()-1, inputs[:,0].max()+1
        y_min, y_max = inputs[:,1].min()-1, inputs[:,1].max()+1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        grid = np.c_[xx.ravel(), yy.ravel()]
        # prediction and visualization
        z = np.dot(alpha * targets, k_func(inputs, grid))
        ax.contour(xx, yy, z.reshape(xx.shape), levels=[0], colors='green')
        ax.set_title(title)

    plt.show()

inputs, targets = generate_noisy_data()
plot_comparison(inputs, targets)
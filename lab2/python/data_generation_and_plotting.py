import numpy as np
import matplotlib.pyplot as plt
import random

def generate_and_plot_data():
    # 1. Set the random seed for reproducibility
    np.random.seed(100)
    
    # 2. Generate Class A
    # Class A consists of two clusters:
    # 10 points around (1.5, 0.5)
    # 10 points around (-1.5, 0.5)
    # Standard deviation is 0.2 for both.
    
    cluster_a1 = np.random.randn(10, 2) * 0.2 + [1.5, 0.5]
    cluster_a2 = np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]
    classA = np.concatenate((cluster_a1, cluster_a2))
    
    # 3. Generate Class B
    # Class B consists of one cluster:
    # 20 points around (0.0, -0.5)
    # Standard deviation is 0.2.
    classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
    
    # 4. Concatenate Inputs
    inputs = np.concatenate((classA, classB))
    
    # 5. Create Targets
    # Class A is encoded as 1, Class B as -1
    targets = np.concatenate((
        np.ones(classA.shape[0]),
        -np.ones(classB.shape[0])
    ))
    
    # 6. Randomly Reorder (Shuffle) the Samples
    N = inputs.shape[0] # Number of rows (samples)
    permute = list(range(N))
    random.seed(100)    # Optional: ensure shuffle is also reproducible
    random.shuffle(permute)
    
    inputs = inputs[permute, :]
    targets = targets[permute]
    
    print(f"Data generation complete.")
    print(f"Total samples: {N}")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")

    # --- Section 6: Plotting ---
    # Visualizing the data to verify the distributions
    plt.figure(figsize=(8, 6))
    
    # Plot Class A (Red) and Class B (Blue)
    # Note: We plot the original class arrays to color them correctly before shuffling,
    # or we can mask the shuffled 'inputs' using 'targets'.
    
    # Using the logic from the text (plotting the original class arrays for simplicity):
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.', label='Class A (+1)')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.', label='Class B (-1)')
    
    plt.axis('equal') # Force same scale on both axes
    plt.title('Linearly Separable Test Data')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save a copy as requested in the text
    plt.savefig('svmplot.pdf') 
    plt.show()

    return inputs, targets

# Run the function
if __name__ == "__main__":
    inputs, targets = generate_and_plot_data()
import numpy as np
import matplotlib.pyplot as plt

def plot_penalized_objective(f, h, r, xlim=(-1, 6), ylim=(-1, 10), resolution=200):
    """
    Plots the objective function f, constraint h, and the penalized objective.
    
    Parameters:
    - f: objective function, takes np.array of shape (2,)
    - h: equality constraint, h(x) = 0
    - r: penalty parameter (float)
    - xlim, ylim: ranges for x and y axis
    - resolution: number of grid points per axis
    """
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    Z_f = np.zeros_like(X)
    Z_penalized = np.zeros_like(X)
    Z_h = np.zeros_like(X)
    
    for i in range(resolution):
        for j in range(resolution):
            x_ij = np.array([X[i, j], Y[i, j]])
            Z_f[i, j] = f(x_ij)
            Z_h[i, j] = h(x_ij)
            Z_penalized[i, j] = f(x_ij) + r * h(x_ij)**2
    
    plt.figure(figsize=(12, 5))

    # Plot original objective
    plt.subplot(1, 3, 1)
    cs1 = plt.contourf(X, Y, Z_f, levels=30, cmap='viridis')
    plt.colorbar(cs1)
    plt.contour(X, Y, Z_h, levels=[0], colors='red', linewidths=2)
    plt.title("Objective Function with Constraint")
    plt.xlabel("x₀")
    plt.ylabel("x₁")
    
    # Plot penalized objective
    plt.subplot(1, 3, 2)
    cs2 = plt.contourf(X, Y, Z_penalized, levels=30, cmap='plasma')
    plt.colorbar(cs2)
    plt.contour(X, Y, Z_h, levels=[0], colors='white', linewidths=2)
    plt.title(f"Penalized Objective (r = {r})")
    plt.xlabel("x₀")
    plt.ylabel("x₁")
    
    # Plot constraint as heatmap (optional visualization)
    plt.subplot(1, 3, 3)
    cs3 = plt.contourf(X, Y, Z_h, levels=30, cmap='coolwarm')
    plt.colorbar(cs3)
    plt.contour(X, Y, Z_h, levels=[0], colors='black', linewidths=2)
    plt.title("Constraint Function h(x)")
    plt.xlabel("x₀")
    plt.ylabel("x₁")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    f = lambda x: (x[0] - 5)**2 + (x[1] - 6)**2
    h = lambda x: x[0] - 2

    plot_penalized_objective(f, h, r=1)
    plot_penalized_objective(f, h, r=10)
    plot_penalized_objective(f, h, r=100)
    plot_penalized_objective(f, h, r=1000)
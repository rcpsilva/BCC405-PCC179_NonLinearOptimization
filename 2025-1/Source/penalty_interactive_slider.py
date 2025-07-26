import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define the objective and constraint
f = lambda x: (x[0] - 5)**2 + (x[1] - 6)**2
h = lambda x: x[0] - 2

def compute_values(f, h, r, X, Y):
    Z_f = np.zeros_like(X)
    Z_penalized = np.zeros_like(X)
    Z_h = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_ij = np.array([X[i, j], Y[i, j]])
            Z_f[i, j] = f(x_ij)
            Z_h[i, j] = h(x_ij)
            Z_penalized[i, j] = f(x_ij) + r * h(x_ij)**2
    return Z_f, Z_penalized, Z_h

def main():
    # Grid setup
    resolution = 200
    x = np.linspace(-1, 6, resolution)
    y = np.linspace(-1, 10, resolution)
    X, Y = np.meshgrid(x, y)

    # Initial penalty value
    r0 = 1.0
    Z_f, Z_penalized, Z_h = compute_values(f, h, r0, X, Y)

    # Plot setup
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    plt.subplots_adjust(bottom=0.25)

    # Plot 1: f(x)
    c1 = axs[0].contourf(X, Y, Z_f, levels=30, cmap='viridis')
    axs[0].contour(X, Y, Z_h, levels=[0], colors='red', linewidths=2)
    axs[0].set_title("Objective Function")
    axs[0].set_xlabel("x₀")
    axs[0].set_ylabel("x₁")

    # Plot 2: Penalized objective
    c2 = axs[1].contourf(X, Y, Z_penalized, levels=30, cmap='plasma')
    constraint_line = axs[1].contour(X, Y, Z_h, levels=[0], colors='white', linewidths=2)
    axs[1].set_title(f"Penalized (r = {r0:.1f})")
    axs[1].set_xlabel("x₀")
    axs[1].set_ylabel("x₁")

    # Plot 3: Constraint
    c3 = axs[2].contourf(X, Y, Z_h, levels=30, cmap='coolwarm')
    axs[2].contour(X, Y, Z_h, levels=[0], colors='black', linewidths=2)
    axs[2].set_title("Constraint h(x)")
    axs[2].set_xlabel("x₀")
    axs[2].set_ylabel("x₁")

    # Slider for r
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    r_slider = Slider(ax_slider, 'r (penalty)', 0.1, 100.0, valinit=r0, valstep=0.5)

    def update(val):
        r = r_slider.val
        _, Z_penalized, _ = compute_values(f, h, r, X, Y)
        axs[1].cla()
        axs[1].contourf(X, Y, Z_penalized, levels=30, cmap='plasma')
        axs[1].contour(X, Y, Z_h, levels=[0], colors='white', linewidths=2)
        axs[1].set_title(f"Penalized (r = {r:.1f})")
        axs[1].set_xlabel("x₀")
        axs[1].set_ylabel("x₁")
        fig.canvas.draw_idle()

    r_slider.on_changed(update)
    plt.show()

if __name__ == "__main__":
    main()

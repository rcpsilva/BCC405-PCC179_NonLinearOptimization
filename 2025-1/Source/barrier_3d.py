import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection

# Objective function
f = lambda x: (x[0] - 5)**2 + (x[1] - 6)**2

# Inequality constraint: g(x) <= 0
g = lambda x: x[0] - 2

def compute_values_barrier(f, g, r, X, Y):
    Z_f = np.zeros_like(X)
    Z_barrier_log = np.zeros_like(X)
    Z_barrier_inv = np.zeros_like(X)
    Z_g = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_ij = np.array([X[i, j], Y[i, j]])
            Z_f[i, j] = f(x_ij)
            Z_g[i, j] = g(x_ij)

            if g(x_ij) < 0:
                # Log barrier: -r * log(-g(x))
                Z_barrier_log[i, j] = f(x_ij) - r * np.log(-g(x_ij))
                # Inverse barrier: -r / g(x)
                Z_barrier_inv[i, j] = f(x_ij) - r / g(x_ij)
            else:
                Z_barrier_log[i, j] = np.nan
                Z_barrier_inv[i, j] = np.nan

    return Z_f, Z_barrier_log, Z_barrier_inv, Z_g


def main():
    resolution = 100
    x = np.linspace(-1, 6, resolution)
    y = np.linspace(-1, 10, resolution)
    X, Y = np.meshgrid(x, y)

    r0 = 1.0
    Z_f, Z_barrier_log, Z_barrier_inv, Z_g = compute_values_barrier(f, g, r0, X, Y)

    # Create figure and axes
    fig = plt.figure(figsize=(18, 6))
    plt.subplots_adjust(bottom=0.25)

    # Plot 1: Objective function (2D contour)
    ax1 = fig.add_subplot(131)
    ax1.contourf(X, Y, Z_f, levels=30, cmap='viridis')
    ax1.contour(X, Y, Z_g, levels=[0], colors='red', linewidths=2)
    ax1.set_title("Objective Function")
    ax1.set_xlabel("x₀")
    ax1.set_ylabel("x₁")

    # Plot 2: Log barrier (3D)
    ax2 = fig.add_subplot(132, projection='3d')
    surf_log = ax2.plot_surface(X, Y, Z_barrier_log, cmap='plasma', edgecolor='none')
    ax2.set_title(f"Log Barrier: -r log(-g(x)) (r={r0:.1f})")
    ax2.set_xlabel("x₀")
    ax2.set_ylabel("x₁")
    ax2.set_zlabel("Value")

    # Plot 3: Inverse barrier (3D)
    ax3 = fig.add_subplot(133, projection='3d')
    surf_inv = ax3.plot_surface(X, Y, Z_barrier_inv, cmap='magma', edgecolor='none')
    ax3.set_title(f"Inverse Barrier: -r 1/g(x) (r={r0:.1f})")
    ax3.set_xlabel("x₀")
    ax3.set_ylabel("x₁")
    ax3.set_zlabel("Value")

    # Slider
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    r_slider = Slider(ax_slider, 'r (barrier)', 0.1, 5.0, valinit=r0, valstep=0.1)

    def update(val):
        r = r_slider.val
        _, Z_barrier_log, Z_barrier_inv, _ = compute_values_barrier(f, g, r, X, Y)

        # Update log barrier surface
        ax2.cla()
        ax2.plot_surface(X, Y, Z_barrier_log, cmap='plasma', edgecolor='none')
        ax2.set_title(f"Log Barrier: -r log(-g(x)) (r={r:.1f})")
        ax2.set_xlabel("x₀")
        ax2.set_ylabel("x₁")
        ax2.set_zlabel("Value")

        # Update inverse barrier surface
        ax3.cla()
        ax3.plot_surface(X, Y, Z_barrier_inv, cmap='magma', edgecolor='none')
        ax3.set_title(f"Inverse Barrier: -r 1/g(x) (r={r:.1f})")
        ax3.set_xlabel("x₀")
        ax3.set_ylabel("x₁")
        ax3.set_zlabel("Value")

        fig.canvas.draw_idle()

    r_slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()

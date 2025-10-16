import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===============================
# Configuração do domínio
# ===============================
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(14, 10))

# Função auxiliar para plotar
def plot_surface(Z, title, pos):
    ax = fig.add_subplot(2, 3, pos, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.view_init(elev=25, azim=45)

    # contorno
    ax2 = fig.add_subplot(2, 3, pos+3)
    ax2.contourf(X, Y, Z, 20, cmap='viridis')
    ax2.set_title("Contorno: " + title)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

# ===============================
# 1) Hessiana Positiva Definida
# f(x,y) = x^2 + y^2
# ===============================
Z1 = X**2 + Y**2
plot_surface(Z1, "PD: x² + y²", 1)

# ===============================
# 2) Hessiana Negativa Definida
# f(x,y) = -x² - y²
# ===============================
Z2 = -X**2 - Y**2
plot_surface(Z2, "ND: -x² - y²", 2)

# ===============================
# 3) Hessiana Positiva Semidefinida
# f(x,y) = x²
# ===============================
Z3 = X**2
plot_surface(Z3, "PSD: x²", 3)

# ===============================
# 4) Hessiana Negativa Semidefinida
# f(x,y) = -x²
# ===============================
Z4 = -X**2
plot_surface(Z4, "NSD: -x²", 4)

# ===============================
# 5) Hessiana Indefinida
# f(x,y) = x² - y²  → forma de sela
# ===============================
Z5 = X**2 - Y**2
plot_surface(Z5, "Indefinida: x² - y²", 5)

plt.tight_layout()
plt.show()

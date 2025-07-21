from optimizers import gradient_descent, gradient_cristiano, gradient_adaptive, gradient_momentum, RMSProp, Adam
from visualize import plot_sequence, function_contour
import numpy as np
import matplotlib.pyplot as plt


fs = []
gs = []


fs.append(lambda x: 0.1*x[0]**2 + 10*x[1]**2)
gs.append(lambda x: np.array([0.2*x[0], 20*x[1]])) #0.09

fs.append(lambda x: 0.1*(x[0]-1)**2 + 10*(x[1]-2)**2)
gs.append(lambda x: np.array([0.2*(x[0]-1), 20*(x[1]-2)]))


fs.append(lambda x: 2*x[0]**2 + x[1]**2 + 2*x[0]*x[1])
gs.append(lambda x: np.array([4*x[0] + 2*x[1], 2*x[1] + 2*x[0]])) #0.03

# Função de Rosenbrock (2 variáveis)
fs.append(lambda x: 100.0 * (x[1] - x[0]**2)**2 + (1.0 - x[0])**2)

# Gradiente da Rosenbrock
gs.append(lambda x: np.array([
    -400.0 * (x[1] - x[0]**2) * x[0] - 2.0 * (1.0 - x[0]),  # ∂f/∂x₁
     200.0 * (x[1] - x[0]**2)                               # ∂f/∂x₂
]))

# Função de Booth (mínimo único em (1, 3) com f = 0)
fs.append(lambda x: (x[0] + 2.0 * x[1] - 7.0)**2 + (2.0 * x[0] + x[1] - 5.0)**2)

# Gradiente da Booth
gs.append(lambda x: np.array([
    10.0 * x[0] +  8.0 * x[1] - 34.0,   # ∂f/∂x₁
     8.0 * x[0] + 10.0 * x[1] - 38.0    # ∂f/∂x₂
]))

x0 = np.array([-4.5,-4.5])
prob_idx = 3
niter = 5000

#_, gd1, f_gd1, _ = gradient_descent(x0,fs[prob_idx],gs[prob_idx],step_size=1e-4,niter=niter,tol=1e-6)
_, ga1, f_ga1, _ = gradient_adaptive(x0,fs[prob_idx],gs[prob_idx],step_size=0.9,niter=niter,tol=1e-6)
#_, gm1, f_gm1, _ = gradient_momentum(x0,fs[prob_idx],gs[prob_idx],step_size=1e-4,niter=niter,tol=1e-6,beta=0)
#_, rms, f_rms, _ = RMSProp(x0,fs[prob_idx],gs[prob_idx],step_size=0.05,niter=niter,tol=1e-6,beta=0.2)
_, ada, f_ada, _ = Adam(x0,fs[prob_idx],gs[prob_idx],step_size=0.9,niter=niter,tol=1e-6,beta1=0.9,beta2=0.99)

res = [(ada, f_ada),(ga1, f_ga1)]

#res = [(ga1,f_ga1)]

ax = function_contour(fs[prob_idx],[-5,-5],[5,5],0.1,30)

for r in res:
    plot_sequence(fs[prob_idx],r[0],ax)

for r in res:
    print(f'niter: {len(r[0])} best: {r[1][-1]}')

plt.show()
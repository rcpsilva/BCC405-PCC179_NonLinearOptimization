from optimizers import gradient_descent, gradient_cristiano, gradient_adaptive, gradient_momentum
from visualize import plot_sequence, function_contour
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: 0.1*x[0]**2 + 10*x[1]**2
g = lambda x: np.array([0.2*x[0], 20*x[1]]) #0.09

#f = lambda x: 2*x[0]**2 + x[1]**2 + 2*x[0]*x[1]
#g = lambda x: np.array([4*x[0] + 2*x[1], 2*x[1] + 2*x[0]]) #0.03

x0 = np.array([-4.5,-4.5])

_, gd1, _, _ = gradient_descent(x0,f,g,step_size=0.09,niter=1000,tol=1e-6)
#_, gc1, _, _ = gradient_cristiano(x0,f,g,step_size=0.05,niter=1000,tol=1e-6)
#_, ga1, _, _ = gradient_adaptive(x0,f,g,step_size=0.5,niter=1000,tol=1e-6)
_, gm1, _, _ = gradient_momentum(x0,f,g,step_size=0.1,niter=1000,tol=1e-6,beta=0.5)

ax = function_contour(f,[-5,-5],[5,5],0.1,30)

plot_sequence(f,gd1,ax)
#plot_sequence(f,gc1,ax)
#plot_sequence(f,ga1,ax)
plot_sequence(f,gm1,ax)

print(f'niter: \n gd1 {len(gd1)} \n gm1 {len(gm1)}')

plt.show()
import numpy as np
import matplotlib.pyplot as plt

def plot_sequence(fun,seq,lb,up,delta,levels):

    xs = np.arange(lb[0], up[0], delta)
    ys = np.arange(lb[1], up[1], delta)

    Z = np.zeros((len(xs),len(ys)))

    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            Z[i][j] = fun([x,y])

    X, Y = np.meshgrid(xs, ys)

    plt.contour(X, Y, Z,levels=levels)
    p0 = [p[0] for p in seq]
    p1 = [p[1] for p in seq]
    plt.plot(p1,p0,'-')

    #for p in seq:
    #    plt.plot(p[0],p[1],'r.')

    

if __name__ == '__main__':

    from optimize import gradient_descent, gradient_descent_adaptive_step

    #f = lambda x: x[0] - x[1] + 2*x[0]**2 + 2*x[0]*x[1] + x[1]**2
    #g = lambda x: np.array([1 + 4*x[0] + 2*x[1], -1 + 2*x[0] + 2*x[1]])

    f = lambda x: 10*x[0]**2 + 2*x[1]**2
    g = lambda x: np.array([20*x[0], 4*x[1]])

    #f = lambda x: 2*x[0]**2 + x[1]**2 + 2*x[0]*x[1]
    #g = lambda x: np.array([4*x[0] + 2*x[1], 2*x[1] + 2*x[0]])

    x0 = np.array([-4,-4])

    _, gd1, _, _ = gradient_descent(x0,f,g,step_size=0.01,niter=1000,tol=1e-6)
    _, gd2, _, _ = gradient_descent(x0,f,g,step_size=0.09,niter=1000,tol=1e-6) 
    _, gda, _, _ = gradient_descent_adaptive_step(x0,f,g,step_size=2,niter=1000,tol=1e-6)

    
    plot_sequence(f,gd1,[-5,-5],[5,5],0.1,30)
    plot_sequence(f,gd2,[-5,-5],[5,5],0.1,30)
    plot_sequence(f,gda,[-5,-5],[5,5],0.1,30)
    plt.show()
    #plt.plot(ys)
    #plt.show()
    #plt.plot(ss)
    #plt.show()
    #print(ss)
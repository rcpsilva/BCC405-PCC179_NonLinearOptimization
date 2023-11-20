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

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z,levels=levels)
    #ax.clabel(CS, inline=False, fontsize=10)
    p0 = [p[0] for p in seq]
    p1 = [p[1] for p in seq]
    plt.plot(p0,p1,'-d')

    #for p in seq:
    #    plt.plot(p[0],p[1],'r.')

    plt.show()

if __name__ == '__main__':

    from optmize import gradient_descent

    f = lambda x: x[0]**2 + x[1]**2 + 1*x[0]*x[1]
    g = lambda x: np.array([2*x[0] + 0.5*x[1], 2*x[1] + 0.5*x[0]])
    x0 = np.array([2,2])

    x, xs, ys, ss = gradient_descent(x0,f,g,step_size=10,niter=500,tol=1e-3)

    x0s = [x[0] for x in xs]
    x1s = [x[1] for x in xs]

    plot_sequence(f,xs,[-2,-2],[2,2],0.1,30)
    plt.plot(ys)
    plt.show()
    plt.plot(ss)
    plt.show()
    print(ss)

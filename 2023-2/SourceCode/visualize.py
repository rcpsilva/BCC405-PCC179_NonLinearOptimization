import numpy as np
import matplotlib.pyplot as plt

def function_contour(fun,lb,up,delta,levels):
    xs = np.arange(lb[0], up[0], delta)
    ys = np.arange(lb[1], up[1], delta)

    Z = np.zeros((len(xs),len(ys)))

    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            Z[i][j] = fun([x,y])

    X, Y = np.meshgrid(xs, ys)

    plt.contour(X, Y, Z,levels=levels)

def plot_sequence(fun,seq,lb,up,delta,levels):

    function_contour(fun,lb,up,delta,levels)

    p0 = [p[0] for p in seq]
    p1 = [p[1] for p in seq]
    plt.plot(p1,p0,'-d')

def plot_directions(fun,M,lb,up,delta,levels):

    function_contour(fun,lb,up,delta,levels)

    

if __name__ == '__main__':

    pass
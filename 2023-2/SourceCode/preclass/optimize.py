import numpy as np 
from copy import copy

def rmsprop(x0,f,grad,niter=500,tol=1e-6,gamma=0.4,eta=0.9):

    x = copy(x0)
    iter = 0
    xs = [x0]
    ys = [f(x0)]
    s = np.zeros(len(x0))
    

    while (iter < niter) and (np.linalg.norm(grad(x),2) > tol):
        
        g = grad(x)
        s = gamma*s + (1-gamma)*g**2      
        x = x - eta/(np.sqrt(s + 1e-12))*g


        xs.append(x)
        ys.append(f(x))
        print(f'[{iter}] f: {f(x):.4f} g: {np.linalg.norm(grad(x),2):.4f} d:{g}')
        iter+=1
        

    return x, xs, ys



def gd_momentum(x0,f,grad,niter=500,tol=1e-6,eta=0.05,beta=0.6):

    x = copy(x0)
    iter = 0
    xs = [x0]
    ys = [f(x0)]
    v = np.zeros(len(x0))


    while (iter < niter) and (np.linalg.norm(grad(x),2) > tol):
        d = -grad(x)
        
        v = beta*v + d        
        x = x + eta*v

        xs.append(x)
        ys.append(f(x))
        print(f'[{iter}] f: {f(x):.4f} g: {np.linalg.norm(grad(x),2):.4f} d:{d}')
        iter+=1
        

    return x, xs, ys


def gradient_descent(x0,f,grad,step_size,niter=500,tol=1e-6,gamma=0.98):

    x = copy(x0)
    iter = 0
    xs = [x0]
    ys = [f(x0)]
    ss = [step_size]
    s = step_size
    while (iter < niter) and (np.linalg.norm(grad(x),2) > tol):
        d = -grad(x)
        x = x + s*d
        ss.append(s)
        xs.append(x)
        ys.append(f(x))
        print(f'[{iter}] f: {f(x):.4f} g: {np.linalg.norm(grad(x),2):.4f} d:{d}')
        iter+=1
        s = step_size
        

    return x, xs, ys,ss

def gradient_descent_adaptive_step(x0,f,grad,step_size,niter=500,tol=1e-6,gamma=0.98):

    x = copy(x0)
    iter = 0
    xs = [x0]
    ys = [f(x0)]
    ss = [step_size]
    s = step_size
    while (iter < niter) and (np.linalg.norm(grad(x),2) > tol):
        d = -grad(x)
        while f(x + s*d) > f(x):
            s = s*gamma
        x = x + s*d
        ss.append(s)
        xs.append(x)
        ys.append(f(x))
        print(f'[{iter}] f: {f(x):.4f} g: {np.linalg.norm(grad(x),2):.4f} d:{d}')
        iter+=1
        s = step_size
        

    return x, xs, ys,ss


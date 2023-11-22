import numpy as np 

def gradient_descent(x0,f,grad,step_size,niter=500,tol=1e-6,gamma=0.98):

    x = x0
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

    x = x0
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


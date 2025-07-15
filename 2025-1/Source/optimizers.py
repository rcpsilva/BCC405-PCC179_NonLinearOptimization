import numpy as np

def Adam(x0,f,grad,step_size=0.9,niter=500,tol=1e-6,beta1=0.9,beta2=0.99):
    eps = 1e-10
    x = x0
    iter = 0
    xs = [x0]
    ys = [f(x0)]
    ss = [step_size]
    s = step_size
    z = grad(x0)**2 #np.zeros_like(x)
    v = grad(x0)
    while (iter < niter) and (np.linalg.norm(grad(x),2) > tol):
        d = grad(x)
        v = beta1*v + (1 - beta1)*d
        z = beta2*z + (1 - beta2)*d**2 
        v_hat = v/(1 - beta1**(iter+1))
        z_hat = z/(1- beta2**(iter+1))
        x = x - s*(v_hat/(np.sqrt(z_hat) + eps))
        ss.append(s)
        xs.append(x)
        ys.append(f(x))
        print(f'[{iter}] f: {f(x):.4f} g: {np.linalg.norm(grad(x),2):.4f} d:{v_hat}')
        iter+=1
        s = step_size
        
    return x, xs, ys, ss


def RMSProp(x0,f,grad,step_size=0.9,niter=500,tol=1e-6,beta=0.98):
    eps = 1e-10
    x = x0
    iter = 0
    xs = [x0]
    ys = [f(x0)]
    ss = [step_size]
    s = step_size
    z = grad(x0)**2 #np.zeros_like(x)
    while (iter < niter) and (np.linalg.norm(grad(x),2) > tol):
        d = grad(x)
        z = beta*z + (1 - beta)*d**2 
        x = x - s*(d/(np.sqrt(z) + eps))
        ss.append(s)
        xs.append(x)
        ys.append(f(x))
        print(f'[{iter}] f: {f(x):.4f} g: {np.linalg.norm(grad(x),2):.4f} d:{d}')
        iter+=1
        s = step_size
        
    return x, xs, ys, ss


def gradient_momentum(x0,f,grad,step_size=0.9,niter=500,tol=1e-6,beta=0.98):

    x = x0
    iter = 0
    xs = [x0]
    ys = [f(x0)]
    ss = [step_size]
    s = step_size
    v = np.zeros_like(x)
    while (iter < niter) and (np.linalg.norm(grad(x),2) > tol):
        d = grad(x)
        v = beta*v + (1-beta)*d
        x = x - s*v
        ss.append(s)
        xs.append(x)
        ys.append(f(x))
        print(f'[{iter}] f: {f(x):.4f} g: {np.linalg.norm(grad(x),2):.4f} d:{d}')
        iter+=1
        s = step_size
        

    return x, xs, ys, ss

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
        

    return x, xs, ys, ss

def gradient_adaptive(x0,f,grad,step_size,niter=500,tol=1e-6,gamma=0.98):

    x = x0
    iter = 0
    xs = [x0]
    ys = [f(x0)]
    ss = [step_size]
    s = step_size
    while (iter < niter) and (np.linalg.norm(grad(x),2) > tol):
        d = -grad(x)
        s,count = find_eta_armijo(f,x,grad,d)
        x = x + s*d
        ss.append(s)
        xs.append(x)
        ys.append(f(x))
        print(f'[{iter}] f: {f(x):.4f} g: {np.linalg.norm(grad(x),2):.4f} d:{d} count:{count}')
        iter+=1
        s = step_size
        

    return x, xs, ys, ss

def find_eta_armijo(f,x,gf,d):

    gamma = 0.9
    eta = 0.5

    l = 1
    count = 0

    while f(x + l*d) > f(x) + eta*l*np.dot(gf(x),d.T):
        l = gamma*l
        count+=1

    return l,count

def gradient_cristiano(x0,f,grad,step_size,niter=500,tol=1e-6,gamma=0.98):

    x = x0
    iter = 0
    xs = [x0]
    ys = [f(x0)]
    ss = [step_size]
    s = step_size
    increment = 0.005
    while (iter < niter) and (np.linalg.norm(grad(x),2) > tol):
        d = -grad(x)
        x = x + s*d
        ss.append(s)
        xs.append(x)
        ys.append(f(x))
        print(f'[{iter}] f: {f(x):.4f} g: {np.linalg.norm(grad(x),2):.4f} d:{d}')
        iter+=1
        s = s + increment
        
    return x, xs, ys, ss
import numpy as np

def gradient_descent(x0,f,gf,eta,i_max,tol,verbose=True):

    i = 0
    x = x0
    while (i < i_max) and (np.linalg.norm(gf(x)) > tol):
        x = x - eta * gf(x)
        i += 1

        if verbose:
            print(f'{i}: \t x: {x} \t ||g||: {np.linalg.norm(gf(x)):.6f} \t f(x): {f(x):.4f}')

def example1():

    f = lambda x : x[0]**2 + x[1]**2 + 20
    gf = lambda x: np.array([2*x[0], 2*x[1]])
    eta = 0.05
    i_max = 1000
    tol = 1e-6
    x0 = np.array([-5,5])
    return [x0,f,gf,eta,i_max,tol]

if __name__ == '__main__':
    
    params = example1()
    gradient_descent(*params)



    


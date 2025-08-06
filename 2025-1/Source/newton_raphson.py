f = lambda x: (x - 5)**2
df = lambda x: 2 * (x - 5)  # derivative of f

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        print(x)
        fx = f(x)
        dfx = df(x)
        if abs(fx) < tol:
            break
        if dfx == 0:
            raise ValueError("Derivative is zero. No convergence.")
        x = x - fx / dfx
    return x

root = newton_raphson(f, df, x0=0)
print("Root:", root)

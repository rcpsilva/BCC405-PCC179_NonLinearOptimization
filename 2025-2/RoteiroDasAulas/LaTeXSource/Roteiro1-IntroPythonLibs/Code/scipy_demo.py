# ============================================
# Nonlinear Optimization - Class Demo Code
# Rosenbrock function (banana) - SciPy optimize
# Unconstrained: Nelder-Mead, BFGS, Newton-CG, trust-ncg
# Constrained: trust-constr, SLSQP, COBYLA
# Paths and contour plots included
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint  # for trust-constr linear constraints

# --------- Objective, Gradient, Hessian (Rosenbrock) ---------
def f_rosen(x):
    # x: array([x, y])
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

def grad_rosen(x):
    # gradient of Rosenbrock
    return np.array([
        -2.0*(1.0 - x[0]) - 400.0*x[0]*(x[1] - x[0]**2),
        200.0*(x[1] - x[0]**2)
    ])

def hess_rosen(x):
    # Hessian of Rosenbrock
    return np.array([
        [2.0 - 400.0*(x[1] - x[0]**2) + 800.0*x[0]**2, -400.0*x[0]],
        [-400.0*x[0], 200.0]
    ])

# --------- Plot utilities ---------
def contour_rosen(ax, xlim=(-2, 2), ylim=(-1, 3), n=400, title="Rosenbrock - contours"):
    xx = np.linspace(xlim[0], xlim[1], n)
    yy = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xx, yy)
    Z = (1.0 - X)**2 + 100.0 * (Y - X**2)**2
    cs = ax.contour(X, Y, Z, levels=np.logspace(-1, 3.5, 20))
    ax.clabel(cs, inline=1, fontsize=7, fmt="%.1e")
    ax.scatter([1.0], [1.0], c='green', marker='*', s=100, label='optimum (1,1)')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title(title)
    return X, Y, Z

def plot_line_constraints(ax, xlim=(-2, 2)):
    # x + y = 1  (dashed)
    x_line = np.linspace(xlim[0], xlim[1], 400)
    ax.plot(x_line, 1.0 - x_line, 'k--', linewidth=1, label='x+y=1')
    # x = y  (solid)
    ax.plot(x_line, x_line, 'k-', linewidth=1, label='x=y')

def plot_halfplane_shade(ax, xlim=(-2,2), ylim=(-1,3), n=400):
    # Shade half-plane x+y >= 1
    xx = np.linspace(xlim[0], xlim[1], n)
    yy = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xx, yy)
    mask = (X + Y >= 1.0).astype(float)
    mask = np.ma.masked_where(mask < 0.5, mask)
    ax.contourf(X, Y, mask, levels=[0.5, 1.5], alpha=0.08)

def plot_path(ax, path, label):
    if len(path) == 0:
        return
    P = np.array(path)
    ax.plot(P[:,0], P[:,1], marker='o', markersize=3, linewidth=1.2, label=label)

# --------- Demo: Unconstrained methods ---------
def demo_unconstrained(x0=np.array([-1.2, 1.0])):
    # callbacks to store paths (accept optional state to be generic)
    path_nm, path_bfgs, path_ncg, path_tncg = [], [], [], []

    def cb_nm(xk, state=None):     path_nm.append(np.copy(xk))
    def cb_bfgs(xk, state=None):   path_bfgs.append(np.copy(xk))
    def cb_ncg(xk, state=None):    path_ncg.append(np.copy(xk))
    def cb_tncg(xk, state=None):   path_tncg.append(np.copy(xk))

    print("\n=== UNCONSTRAINED ===")
    print("Start x0 =", x0)

    # Nelder-Mead (no derivatives)
    res_nm = minimize(f_rosen, x0, method='Nelder-Mead',
                      callback=cb_nm,
                      options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-12})
    print("[Nelder-Mead] x* =", res_nm.x, " f(x*) =", res_nm.fun, " success=", res_nm.success)

    # BFGS (with gradient)
    res_bfgs = minimize(f_rosen, x0, method='BFGS',
                        jac=grad_rosen, callback=cb_bfgs,
                        options={'gtol': 1e-8, 'maxiter': 1000})
    print("[BFGS]        x* =", res_bfgs.x, " f(x*) =", res_bfgs.fun, " success=", res_bfgs.success)

    # Newton-CG (with grad + Hessian)
    res_ncg = minimize(f_rosen, x0, method='Newton-CG',
                       jac=grad_rosen, hess=hess_rosen, callback=cb_ncg,
                       options={'xtol': 1e-12, 'maxiter': 1000})
    print("[Newton-CG]   x* =", res_ncg.x, " f(x*) =", res_ncg.fun, " success=", res_ncg.success)

    # trust-ncg (trust-region Newton CG) with grad + Hessian
    res_tncg = minimize(f_rosen, x0, method='trust-ncg',
                        jac=grad_rosen, hess=hess_rosen,
                        callback=cb_tncg,
                        options={'gtol': 1e-8, 'maxiter': 1000})
    print("[trust-ncg]   x* =", res_tncg.x, " f(x*) =", res_tncg.fun, " success=", res_tncg.success)

    # Plot
    fig, ax = plt.subplots(figsize=(6,5))
    contour_rosen(ax, title="Unconstrained: contours and paths")
    ax.scatter([x0[0]], [x0[1]], c='red', marker='x', s=60, label='start')
    plot_path(ax, path_nm,   'Nelder-Mead')
    plot_path(ax, path_bfgs, 'BFGS')
    plot_path(ax, path_ncg,  'Newton-CG')
    plot_path(ax, path_tncg, 'trust-ncg')
    ax.legend(loc='upper right', fontsize=7)
    plt.tight_layout()
    plt.show()

# --------- Demo: Constrained methods ---------
def demo_constrained(x0=np.array([0.8, 0.8])):
    # Constraints:
    #   x + y >= 1  (linear)
    #   x - y  = 0  (linear)
    A  = np.array([[1.0,  1.0],   # x + y
                   [1.0, -1.0]])  # x - y
    lb = np.array([1.0, 0.0])     # [ >=1, =0 ]
    ub = np.array([np.inf, 0.0])  # [ inf,  0 ]
    lincon = LinearConstraint(A, lb, ub)

    # Dict-style constraints (for SLSQP)
    cons_slsqp = [
        {'type': 'ineq', 'fun': lambda z: z[0] + z[1] - 1.0},  # x+y-1 >= 0
        {'type': 'eq',   'fun': lambda z: z[0] - z[1]}         # x - y = 0
    ]

    # COBYLA only accepts 'ineq', approximate x=y by two ineqs
    cons_cobyla = [
        {'type': 'ineq', 'fun': lambda z: z[0] + z[1] - 1.0},  # x+y-1 >= 0
        {'type': 'ineq', 'fun': lambda z:  z[0] - z[1]},       # x - y >= 0
        {'type': 'ineq', 'fun': lambda z:  z[1] - z[0]}        # y - x >= 0
    ]

    # callbacks to store paths (accept optional state)
    path_tc, path_slsqp, path_cobyla = [], [], []

    def cb_tc(xk, state=None):
        path_tc.append(np.copy(xk))
        # you can also inspect 'state' (e.g., state.grad_norm) if desired

    def cb_slsqp(xk, state=None):
        path_slsqp.append(np.copy(xk))

    def cb_cobyla(xk, state=None):
        path_cobyla.append(np.copy(xk))

    print("\n=== CONSTRAINED (x+y>=1, x=y) ===")
    print("Start x0 =", x0)

    # trust-constr (with LinearConstraint)
    res_tc = minimize(f_rosen, x0, method='trust-constr',
                      jac=grad_rosen, hess=hess_rosen,
                      constraints=[lincon],
                      callback=cb_tc,
                      options={'maxiter': 300, 'gtol': 1e-8, 'xtol': 1e-12})
    print("[trust-constr] x* =", res_tc.x, " f(x*) =", res_tc.fun, " success=", res_tc.success)

    # SLSQP (dict constraints)
    res_slsqp = minimize(f_rosen, x0, method='SLSQP',
                         jac=grad_rosen, constraints=cons_slsqp,
                         callback=cb_slsqp,
                         options={'maxiter': 300, 'ftol': 1e-12})
    print("[SLSQP]        x* =", res_slsqp.x, " f(x*) =", res_slsqp.fun, " success=", res_slsqp.success)

    # COBYLA (ineq only; no derivatives)
    res_cobyla = minimize(f_rosen, x0, method='COBYLA',
                          constraints=cons_cobyla,
                          callback=cb_cobyla,
                          options={'maxiter': 1000, 'rhobeg': 0.5, 'tol': 1e-6})
    print("[COBYLA]       x* =", res_cobyla.x, " f(x*) =", res_cobyla.fun, " success=", res_cobyla.success)

    # Plot with constraints visuals
    fig, ax = plt.subplots(figsize=(6,5))
    contour_rosen(ax, title="Constrained: region and paths")
    plot_line_constraints(ax)
    plot_halfplane_shade(ax)
    ax.scatter([x0[0]], [x0[1]], c='red', marker='x', s=60, label='start')
    plot_path(ax, path_tc,     'trust-constr')
    plot_path(ax, path_slsqp,  'SLSQP')
    plot_path(ax, path_cobyla, 'COBYLA')
    ax.legend(loc='upper right', fontsize=7)
    plt.tight_layout()
    plt.show()

# --------- Main ---------
if __name__ == "__main__":
    # Classical start for unconstrained demo
    x0_unc = np.array([-1.2, 1.0])
    demo_unconstrained(x0=x0_unc)

    # Feasible start for constrained demo (x+y>=1 and x=y)
    x0_constr = np.array([0.8, 0.8])
    demo_constrained(x0=x0_constr)

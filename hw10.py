#Math 260 Homework 10
#author: Max Wang

import matplotlib.pyplot as plt
import numpy as np


def rk4(func, interval, y0, h):
    """ RK4 method, to solve the system y' = f(t,y) of m equations
        Inputs:
            f - a function returning an np.array of length m
         a, b - the interval to integrate over
           y0 - initial condition (a list or np.array), length m
           h  - step size to use
        Returns:
            t - a list of the times for the computed solution
            y - computed solution; list of m components (len(y[k]) == len(t))
    """
    y = np.array(y0)
    t = interval[0]
    tvals = [t]
    yvals = [[v] for v in y]  # y[j][k] is j-th component of solution at t[k]
    while t < interval[1] - 1e-12:
        f0 = func(t, y)
        f1 = func(t + 0.5*h, y + 0.5*h*f0)
        f2 = func(t + 0.5*h, y + 0.5*h*f1)
        f3 = func(t + h, y + h*f2)
        y += (1.0/6)*h*(f0 + 2*f1 + 2*f2 + f3)
        t += h
        for k in range(len(y)):
            yvals[k].append(y[k])
        tvals.append(t)

    return tvals, yvals


def read_sir_data(fname):
    """ Reads the SIR data for the HW problem.
        Inputs:
            fname - the filename (should be "sir_data.txt")
        Returns:
            t, x - the data (t[k], I[k]), where t=times, I= # infected
            pop - the initial susceptible population (S(0))
    """
    with open(fname, 'r') as fp:
        parts = fp.readline().split()
        pop = float(parts[0])
        npts = int(float(parts[1]))
        t = np.zeros(npts)
        x = np.zeros(npts)

        for k in range(npts):
            parts = fp.readline().split()
            t[k] = float(parts[0])
            x[k] = float(parts[1])

    return t, x, pop


def sir_ode(t, x, params):
    """ ODE for the SIR model
        x = [s, i, r]
        params = [alpha, beta]
    """
    s = x[0]
    i = x[1]
    r = x[2]
    p = s + i + r
    a = params[0]
    b = params[1]
    ds = -a*s*i/p
    di = a*s*i/p-b*i
    dr = b*i
    return np.array((ds, di, dr))

def lse(func, r, fname, m = 3):
    """ Calculates the least-squares error
        func: model function
        r: fitting parameters
        fname: name of the file containing the sample data
        m: optional parameter for calculating the rk4 step size
    """
    
    t_data, x_data, pop = read_sir_data(fname)
    step_data = t_data[1] - t_data[0] # spacing between data points
    h = step_data/2**m
    
    interval = [t_data[0], t_data[-1]]
    ic = [pop, x_data[0], 0] # initial condition of the sir model
    tvals, yvals = rk4(lambda t, x: sir_ode(t, x, r), interval, ic, h)
    ivals = yvals[1] # population of infected individuals from model
    
    err_tot = 0
    for j in range(len(t_data)): # go through all the data points
        j_model = 2**m*j # sample every 2^m data points for the rk4 result
        err_tot += (x_data[j]-ivals[j_model])**2
    return err_tot
           

def minimize(f, x0, delta = 0.001, tol=1e-4, maxsteps=100, verb=False):
    """ Gradient descent with a *very simple* line search
        Inputs:
            f - the function f(x)
            df - gradient of f(x)
            x0 - initial point
    """
    x = np.array(x0)
    err = 100
    it = 0
    pts = [np.array(x)]
    
    while err > tol:
        fx = f(x)
        v = np.array(((f([x[0]+delta, x[1]])-f([x[0]-delta, x[1]]))/(2*delta),
                      (f([x[0], x[1]+delta])-f([x[0], x[1]-delta]))/(2*delta)))
        v /= sum(np.abs(v))
        alpha = 1
        while ((x - alpha*v) < 0).any() or (alpha > 1e-10 and f(x - alpha*v) >= fx):
            alpha /= 2
            
        if verb:
            print(f"it={it}, x[0]={x[0]:.8f}, f={fx:.8f}")
        x -= alpha*v
        pts.append(np.array(x))
        err = max(np.abs(alpha*v))
        it += 1

    return x, it, pts



if __name__ == '__main__':

    # load data
    fname = 'sir_data.txt'
    tvals, data, pop = read_sir_data(fname)

    # example ODE solve after picking some parameters
    r0 = [0.5, 0.5]
    
    interval = [tvals[0], tvals[-1]]
    lambda r: lse(sir_ode, r, fname)
    x0 = np.array((pop, data[0], 0))
    r, it, pts = minimize(lambda r: lse(sir_ode, r, fname), r0)
    
    step_data = tvals[1] - tvals[0] # spacing between data points
    h = step_data/2**3
    t, x = rk4(lambda t, x: sir_ode(t, x, r),interval, x0, h)

    print(f'The infection rate, alpha, is: {r[0]}')
    plt.figure(1)
    plt.plot(t, x[1], '-r', tvals, data, '.k', markersize=12)
    plt.xlabel('t')
    plt.ylabel('Infected Population')
    plt.legend(['I(t)', 'Data Points'])
    plt.show()
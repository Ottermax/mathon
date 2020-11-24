#Math 260 Final Project
#author: Max Wang

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

def func1(x):
    return (x+4)*(x+3)*(x-1)*(x-3)/14

def func_plot(f):
    x_vals = np.linspace(-5, 5, 1000)
    f_vals = f(x_vals)
    plt.plot(x_vals, f_vals, 'k-')

def sim_anneal(f, x0, T0, x_it = 20, tol = 1e-3):
    x_vals = [0]
    f_vals = [0]
    T_vals = [[0, 0]] # Temperature and the final f_value at this temperature

    T_trial = 0
    x_trial = 0
    tot_trial = 0
    x_vals[0] = x0
    f_vals[0] = f(x0)
    T_vals[0] = [T0, x0, f(x0)]
    x_opt = x0
    f_opt = f(x0)
    opt_index = 0
    
    terminate = False
    
    while not terminate:
        T_current = T_vals[T_trial][0]
        step_size = 1
        
        while x_trial < x_it:    
            x_current = x_opt
            f_current = f_opt
            
            step = random.uniform(-1, 1)*step_size
            x_new = x_current + step
            f_new = f(x_new)
            
            df = f_new - f_current
        
            if df < 0:
                x_trial += 1
                tot_trial += 1
                x_vals.append(x_new)
                f_vals.append(f_new)
                x_opt = x_new
                f_opt = f_new
                opt_index = tot_trial
            else:
                chance = np.exp(-df/T_current)
                p = random.uniform()
                if p < chance:
                    x_trial += 1
                    tot_trial += 1
                    x_vals.append(x_new)
                    f_vals.append(f_new)
                    
        x_trial = 0
        T_vals[T_trial][1] = x_vals[tot_trial]
        T_vals[T_trial][2] = f_vals[tot_trial]
        
        if len(T_vals) > 3:
            dT_opt = abs(T_vals[T_trial][2] - f_opt)
            dT_1 = abs(T_vals[T_trial][2] - T_vals[T_trial-1][2])
            dT_2 = abs(T_vals[T_trial][2] - T_vals[T_trial-2][2])
            dT_3 = abs(T_vals[T_trial][2] - T_vals[T_trial-3][2])
            if dT_opt < tol and dT_1 < tol and dT_2 < tol and dT_3 < tol:
                terminate = True
            else:
                T_trial += 1
                T_new = 0.9*T_current
                T_vals.append([T_new, x_opt, f_opt])
        else:
            T_trial += 1
            T_new = 0.9*T_current
            T_vals.append([T_new, x_opt, f_opt])
            
    return T_vals, len(T_vals)
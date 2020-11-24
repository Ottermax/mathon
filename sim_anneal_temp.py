#Math 260 Final Project
#author: Max Wang

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter 

def func1(x):
    return (x-15)*(x-21)*(x-11)*(x+11)*(x+5)*(x+2)*(x-3)*(x-6)/10000000

def func_plot(f):
    x_vals = np.linspace(-15, 25, 1000)
    f_vals = f(x_vals)
    plt.plot(x_vals, f_vals, 'k-')
    
def vals_plot(x):
    trials = list(range(len(x)))
    plt.plot(trials, x)

def sim_anneal(f, x0, T0, x_iterations = 10, step_iterations = 10, tol = 1e-4):
    """    
    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    x0 : TYPE
        DESCRIPTION.
    T0 : TYPE
        DESCRIPTION.
    x_it : TYPE, optional
        DESCRIPTION. The default is 10.
    step_it : TYPE, optional
        DESCRIPTION. The default is 10.
    tol : TYPE, optional
        DESCRIPTION. The default is 1e-4.

    Returns
    -------
    T_vals : TYPE
        DESCRIPTION.
    x_vals : TYPE
        DESCRIPTION.
    step_sizes : TYPE
        DESCRIPTION.
    x_opt : TYPE
        DESCRIPTION.
    f_opt : TYPE
        DESCRIPTION.
    """
    
    # initialize lists containing x, f, T, and step size values.
    # lengths indeterminate
    x_vals = [0]         # contains all accepted x
    f_vals = [0]         # f(x_vals)
    T_vals = [[0, 0, 0]] # list of [T, x_T, f_T]
    step_sizes = [0]     # for discussion purposes

    # initialize counters
    x_index = 0          # counter of accepted moves, under fixed T and step size
    T_index = 0          # counter of T
    tot_index = 0        # counter of cumulated accepted moves
    
    # populate lists with initial values
    x_vals[0] = x0
    f_vals[0] = f(x0)
    T_vals[0] = [T0, x0, f(x0)]
    x_opt = x0          # x value of the optimum, initialized to x0
    f_opt = f(x0)       # f value of the optimum, initialized to f(x0)
    
    terminate = False   # terminating condition, initialized to False
    
    
    # loops through temperatures 
    # until the terminating condition has been reached
    while not terminate:
        T_current = T_vals[T_index][0]     # get current temperature
        step_size = 1                      # set the initial step size
        
        # loops through different step sizes
        # for (step_iterations) times
        for step_index in range(step_iterations):
            
            # to adjust step size, need to compare the no. of accepted moves
            # with the no. of all tentative moves under the current step size
            
            attempts = 0   # no. of tentative moves
            
            # loops through different random moves
            # until (x_iterations) of random moves have been accepted
            # uses while-loop because some moves might be rejected
            # so the total number of iterations is indeterminate
            while x_index < x_iterations:   
                attempts += 1                     # count one tentative move
                x_current = x_vals[tot_index]     # get current x
                f_current = f_vals[tot_index]     # get current f
                
                step = random.uniform(-1, 1)*step_size # generate random step
                x_new = x_current + step          # generate new x
                f_new = f(x_new)                  # generate new f
                df = f_new - f_current            # change in "energy"
                
                if df < 0:
                    # always accepts the move if energy is lower
                    x_index += 1                 # count one accepted move
                    tot_index += 1               # count one accepted move
                    x_vals.append(x_new)
                    f_vals.append(f_new)
                    
                    # update the optimum because df < 0
                    x_opt = x_new               
                    f_opt = f_new
                else:
                    # even if df > 0, chance exists to accept the move
                    chance = np.exp(-df/T_current) # Metropolis
                    p = random.uniform()           # random number in [0, 1]
                    if p < chance:
                        # accepts the move if probability falls in chance
                        # the optimum is not updated because df > 0
                        x_index += 1               # count one accepted move
                        tot_index += 1             # count one accepted move
                        x_vals.append(x_new)
                        f_vals.append(f_new)
                      
            # after (x_iterations) of moves have been accepted
            # need to adjust step size and reset counter for next step size
            # success_ratio = no. accepted moves/no. tentative moves
            success_ratio = x_iterations/attempts 
            factor = success_ratio/0.5            # compared with the ideal 0.5
            step_size *= factor                   # change step_size
            step_sizes.append(step_size)
    
            x_index = 0                           # reset counter 
        
        # after (step_iterations) of step size changes have been made
        # need to:
        # a. record the optimum under the current temperature
        # b. check the terminating condition
        # if satisfied --> done
        # otherwise need to reduce temperature and reset counter for the next T
        
        T_vals[T_index][1] = x_opt                # record current optimal x
        T_vals[T_index][2] = f_opt                # record current optimal f
        
        if len(T_vals) > 3: # make sure steady state
            f_current = T_vals[T_index][2]        # get current f
            f_one_before = T_vals[T_index-1][2]   # f for previous T
            f_two_before = T_vals[T_index-2][2]   # f for prev. prev. T
            f_thr_before = T_vals[T_index-3][3]   # f for prev. prev. prev. T
            dT_opt = abs(T_vals[T_index][2] - f_opt)
            dT_1 = abs(T_vals[T_index][2] - T_vals[T_index-1][2])
            dT_2 = abs(T_vals[T_index][2] - T_vals[T_index-2][2])
            dT_3 = abs(T_vals[T_index][2] - T_vals[T_index-3][2])
            if dT_opt < tol and dT_1 < tol and dT_2 < tol and dT_3 < tol:
                terminate = True
            else:
                T_index += 1
                T_new = 0.9*T_current
                T_vals.append([T_new, x_opt, f_opt])
                tot_index += 1
                x_vals.append(x_opt)
                f_vals.append(f_opt)
        else:
            T_index += 1
            T_new = 0.9*T_current
            T_vals.append([T_new, x_opt, f_opt])
            tot_index += 1
            x_vals.append(x_opt)
            f_vals.append(f_opt)
            
        step_index = 0                           # reset counter 
            
    return T_vals, x_vals, step_sizes, x_opt, f_opt
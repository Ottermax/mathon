# Math 260 Final Project
# author: Max Wang

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

def func1(x):
    return (x-13)*(x-8)*(x+10)*(x-1)/1000

def func2(x):
    return (x+1)*(x+2)*x*(x+3)*(x+4)*(x-1)*(x-2)*(x-3.05)/200

def sim_anneal(f, x0, T0, r_T = 0.95, x_iterations = 10, step_iterations = 10, tol = 1e-6):
    """    
    A simplified annealing method
    for finding the global minimum of an 1D function
    
    Parameters
    ----------
    f : function f(x) for which the minimum needs to be found
    x0 : initial independent variable, x
    T0 : initial temperature T
    r_T : rate at which temperature decreases between successive turns
    x_iterations : number of moves to accept for fixed step size and T
                   The default is 10.
    step_iterations : number of step size adjustments to make for fixed T
                      The default is 10.
    tol : tolerance for determining steady-state condition
    
    Returns
    -------
    T_vals : list of [T_k, x_opt_k, f_opt_k]
             showing the k-th temperature, k-th optimal x, and k-th optimal f
    x_vals : list of all accepted moves
    f_vals : list of f for all accepted moves
    x_opt : x of the located global minimum
    f_opt : f of the located global minimum
    step_sizes : list of all step sizes taken
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
        # b. reset counter
        # c. check the terminating condition
        # if satisfied --> done
        # otherwise need to reduce temperature and reset counter for the next T
        
        T_vals[T_index][1] = x_opt                # record current optimal x
        T_vals[T_index][2] = f_opt                # record current optimal f
        
        step_index = 0                           # reset counter 
        
        if len(T_vals) > 3: # make sure steady state
            f_current = T_vals[T_index][2]        # get current f
            f_one_before = T_vals[T_index-1][2]   # f for previous T
            f_two_before = T_vals[T_index-2][2]   # f for prev. prev. T
            f_thr_before = T_vals[T_index-3][2]   # f for prev. prev. prev. T
            
            # calculate improvements compare with previous temperatures
            df_1 = abs(f_current - f_one_before)
            df_2 = abs(f_current - f_two_before)
            df_3 = abs(f_current - f_thr_before)
            if df_1 < tol and df_2 < tol and df_3 < tol:
                # exit loop if no major improvement can be observed
                terminate = True
            else:
                # otherwise, proceed to adjust T
                T_new = 0.9*T_current                  # get new temperature
                T_index += 1                           # count new temperature
                T_vals.append([T_new, x_opt, f_opt])   # update T list
                
                # artificially set the current accepted move to the optimum
                # by adding an accepted move
                # Rationale: approximations for the new T need to start
                # from the current optimum, but without modification,
                # the current accepted move might not be the current optimum
                # i.e. can be a less ideal but still accepted Metropolis move
                tot_index += 1
                x_vals.append(x_opt)
                f_vals.append(f_opt)
        else:
            # keep looping if not in steady state
            # same modifications as above
            T_new = r_T*T_current                  
            T_index += 1                           
            T_vals.append([T_new, x_opt, f_opt]) 
            tot_index += 1
            x_vals.append(x_opt)
            f_vals.append(f_opt)
            
    return T_vals, x_vals, f_vals, x_opt, f_opt, step_sizes

if __name__ == "__main__":
    x_lin = np.linspace(-50, 50, 10000)
    
    # test function 1
    
    #run 10 times:
    xo1_all = [0]*10
    fo1_all = [0]*10
    ss1_all = [0]*10
    for i in range(10):
        Ts1, xs1, fs1, xo1, fo1, ss1 = sim_anneal(func1, 12, 1000)
        xo1_all[i] = xo1
        fo1_all[i] = fo1
        ss1_all[i] = ss1
    
    plt.figure(1)
    f1_lin = func1(x_lin)
    plt.plot(x_lin, f1_lin, 'r-')
    plt.plot(xo1_all, fo1_all, 'ko')
    plt.xlabel('$x$')
    plt.ylabel('$f_1(x)$')
    plt.legend(['$f_1(x)$', 'Detected Minima'])
    plt.title('$f_1(x)$ vs. $x$, with the results of 50 trials ($x_0$ = 12, $T_0$ = 1000)')
    plt.xlim([-14, 18])
    plt.ylim([-10, 15])
    plt.savefig('test_func1.jpg')
    plt.show()
    
    
    
    
    # test function 2
    
    # run 10 times:
    xo2_all = [0]*10
    fo2_all = [0]*10
    ss2_all = [0]*10
    for i in range(10):
        Ts2, xs2, fs2, xo2, fo2, ss2 = sim_anneal(func2, 80, 1000)
        xo2_all[i] = xo2
        fo2_all[i] = fo2
        ss2_all[i] = ss2
        
    plt.figure(2)
    f2_lin = func2(x_lin)
    plt.plot(x_lin, f2_lin, 'r-')
    plt.plot(xo2_all, fo2_all, 'ko')
    plt.xlabel('$x$')
    plt.ylabel('$f_2(x)$')
    plt.legend(['$f_2(x)$', 'Detected Minima'])
    plt.title('$f_2(x)$ vs. $x$, with the results of 50 trials ($x_0$ = 80, $T_0$ = 1000)')
    plt.xlim([-5, 5])
    plt.ylim([-4, 2])
    plt.savefig('test_func2.jpg')
    plt.show()
    
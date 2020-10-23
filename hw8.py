#Math 260 Homework 8
#author: Max Wang

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.random import rand
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

#Q1
def stationary(pt, alpha = 1, steps = 100):
    pt = pt.tocsr() #use csr for fast matrix vector products
    pt_mod = pt.multiply(alpha)
    n = pt_mod.shape[0]
    x = rand(n)
    x /= sum(x)
    for it in range(steps):
        x = pt_mod.dot(x) + (1-alpha)/n*np.full(n, sum(x))
        x /= sum(x)
    return x


def read_graph_file(file):
    fp = open(file, 'r')
    lines = fp.readlines()
    adj = {}
    names = {}
    test = 0
    for line in lines:
        line = line.strip('\n')
        if line:
            parts = line.split(' ')
            node = int(parts[1])
            if parts[0] == 'n':
                names[node] = parts[2]
            elif parts[0] == 'e':
                if node in adj:
                    adj[node].append(int(parts[2]))
                else:
                    adj[node] = [int(parts[2])]         
    fp.close
    return adj, names


def file_to_pt(file):
    adj, names = read_graph_file(file)
    n = len(names)
    val = []
    row = []
    col = []
    for node in adj:
        paths = len(adj[node]) # number of connections
        prob = 1/paths # assuming equal probability to each neighbor
        val.extend([prob]*paths) # add to values
        row.extend([node]*paths) # row is the same for a given node
        col.extend(adj[node]) # cols are the value of the dictionary key
    return coo_matrix((val, (row, col)), shape = (n, n))


def tester_Q1(file, n = 10): # n = number of top choices to be listed
    print('Rank    Score       Website')
    probs = list(stationary(file_to_pt(file)))
    names = read_graph_file(file)[1]
    for i in range(n):
        score = max(probs)
        fav = probs.index(score)
        probs[fav] = 0
        print(f'{i+1: >4}    {score: >.6f}    {names[fav]}')
        
def rk4(f, t, b, y0, h = 0.1):
    y = y0    
    tvals = [t]
    yvals = [y]
    while t < b:
        f1 = f(t, y)
        f2 = f(t + h/2, y + h/2*f1)
        f3 = f(t + h/2, y + h/2*f2)
        f4 = f(t + h, y + h*f3)
        y += h/6*(f1 + 2*f2 + 2*f3 + f4)
        t += h
        tvals.append(t)
        yvals.append(y)
    return tvals, yvals

def func1(t, y):
    return -20*(y-np.sin(t))+np.cos(t)

def sol1(t):
    return np.exp(-20*t)+np.sin(t)

def tester_Q2(h = 0.05):
    #plotting rk4 result vs. exact solution
    ts, ys = rk4(func1, 0, 2, 1, h)
    tsols = np.linspace(0, 2, 1000)
    sols = [sol1(t) for t in tsols]
    plt.plot(ts, ys, 'r--')
    plt.plot(tsols, sols, 'k-')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()
    
    #error analysis
    n = 400 # number of h to analyze
    global_errs = [0]*n
    hs = np.linspace(0.01, 0.13, n)
    for i in range(n):
        ts, ys = rk4(func1, 0, 2, 1, hs[i])
        errs = [abs(sol1(ts[step])-ys[step]) for step in range(len(ts))]
        global_errs[i] = max(errs)
    plt.loglog(hs, global_errs, 'b-')
    plt.loglog(hs, hs**4, 'r--') # reference line
    plt.legend(['max err', 'slope = 4'])
    plt.xlabel('log(h)')
    plt.ylabel('log(err)')
    plt.show()

#Q3
def trap_circ(t = 0, b = 10000*np.pi, x0 = -1, y0 = 0, h = 0.1):
    tvals = [t]
    x = x0
    y = y0
    xvals = [x]
    yvals = [y]
    while t < b:
        x, y = ((1-h**2/4)*x + h*y)/(1+h**2/4), ((1-h**2/4)*y - h*x)/(h**2/4+1)
        t += h
        xvals.append(x)
        yvals.append(y)
        tvals.append(t)
    return tvals, xvals, yvals
    
if __name__ == '__main__':
    #Q1 test case
    #tester_Q1(file)   
    
    #Q2 test case
    tester_Q2()
    
    #Q3 plot
    t, x, y = trap_circ()
    plt.plot(x, y, 'k-')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    #The approximation traces the circle extremely well
    #and can handle a time of a least 10000 pi. 
    #It is safe to say that the approximation will never diverge
    #from the real solution.
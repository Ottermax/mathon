#Math 260 Homework 7
#author: Max Wang

import numpy as np
from numpy.random import uniform, rand
import matplotlib.pyplot as plt

# Q1
def power_method(a, steps, v):
    """ simple implementation of the power method, using a fixed
        number of steps. Computes the largest eigenvalue and
        associated eigenvector.
        Args:
            a - the (n x n) matrix
            steps - the number of iterations to take
        Returns:
            x, r - the eigenvector/value pair such that a*x = r*x
    """
    n = a.shape[0]
    x = rand(n)
    err_vals = [0]*steps
    vee = v/np.sqrt(v.dot(v))
    it = 0
    while it < steps:  # other stopping conditions would go here
        q = np.dot(a, x)  # compute a*x
        r = x.dot(q)    # Rayleigh quotient x_k dot Ax_k / x_k dot x_k
        x = q/np.sqrt(q.dot(q))  # normalize x to a unit vector
        err = abs(x) - abs(vee)
        err_vals[it] = np.sqrt(err.dot(err))
        it += 1
    err_plot(err_vals, np.sqrt(v.dot(v)))
    return x, r

def err_plot(errs, scale):
    # The error converges at a rate of (2/3)^steps.
    # Here, 2 is the second largest eigenvalue, and 3 is the largest eigenvalue.
    # A semilog plot is needed.
    # Red curve is for reference.
    steps = [i for i in range(len(errs))]
    vals = [(2/3)**(step) for step in steps[1:]]
    plt.semilogy(steps[1:], vals, '--r')
    plt.semilogy(steps, errs, '--b')
    plt.ylabel('$\log(err)$')
    plt.xlabel('steps')
    plt.show()
    
# Q2
def stationary(pt, tol = 1e-5):
    max_steps = 500
    x = rand(pt.shape[0])  # random initial vector
    x /= sum(x)
    it = 0
    while it < max_steps: 
        x_old = x
        x = np.dot(pt, x)
        x /= sum(x)
        diff = x_old - x
        diff_size = np.sqrt(diff.dot(diff))
        if diff_size <= tol:
            break
        else: 
            it += 1
    return x, it

def stationary1(pt, steps):
    """Power method to find stationary distribution.
       Given the largest eigenvalue is 1, finds the eigenvector.
    """
    x = rand(pt.shape[0])  # random initial vector
    x /= sum(x)
    for it in range(steps):
        x = np.dot(pt, x)
    return x


def pagerank_small():
    n = 5
    p_matrix = np.array([[0, 1/3, 1/3, 1/3, 0],
                         [1/2, 0, 0, 0, 1/2],
                         [1/2, 1/2, 0, 0, 0],
                         [1/2, 1/2, 0, 0, 0],
                         [0, 1, 0, 0, 0]])
    pt_mod = p_matrix
    
    alpha = 0.95
    for i in range(5):
        for j in range(i+1):
            pt_mod[i][j], pt_mod[j][i] = alpha*pt_mod[j][i]+(1-alpha)/n, alpha*pt_mod[i][j]+(1-alpha)/n

    dist, it = stationary(pt_mod)
    print('Stationary distribution is', dist)
    print('Most popular page is', dist.argmax())
    print('Number of iterations is', it)
    # The highest rank remains the same regardless of changes in alpha (within machine epsilon)
    
# Q3
def read_graph(file):
    fp = open(file, 'r')
    lines = fp.readlines()
    adj = {}
    names = {}
    for line in lines:
        if line[0] == 'n':
            names[int(line[2])] = line[4]
    for each in names:
        adj[each] = []
    for line in lines:
        if line[0] == 'e':
            adj[int(line[2])].append(line[4])
            
    fp.close
    return adj, names

if __name__ == "__main__":   # example from lecture (2x2 matrix)
    a = np.array([[0, 1, 0], [0, 0, 1], [6, -11, 6]])
    v = np.array([1, 3, 9])
    evec, eig = power_method(a, 100, v)
    
    
    np.set_printoptions(precision=3)  # set num. of displayed digits
    print(f"Largest eigenvalue: {eig:.2f}")
    print("Eigenvector: ", evec)
    
    pagerank_small()
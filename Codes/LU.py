#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 15:07:24 2020

@author: selmabeateovland
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg 
import time

start_time = time.time() #to calculate the time in seconds for the program to run

def f(x):   #source term
    return 100.0*np.exp(-10.0*x) 

def anal(x):   #analytical sol
    return 1.0-(1.0-np.exp(-10.0))*x-np.exp(-10.0*x)

def x(n):     # x to be used in analytical sol., ftilde() and f(x)
    x = np.zeros(n+1)   #n+1 because want endpoints for the analytical sol, and len(exact)=len(u)
    h = 1.0/(n+1) #step size
    for i in range(0,n+1):  
        x[i] = i*h
    return x

def exact(n):   #analytical sol for x
    exact = np.zeros(n+1)
    xi = x(n)
    for i in range(0,n+1):
        exact[i] = anal(xi[i])
    return exact

def lu_dec(n):
    A = np.zeros((n+1, n+1)) #our matrix
    xi = x(n)
    h = 1.0/(n+1)   #could probably include this another way? Since we already defined it in another func 
    hh = h*h    #faster
    ftilde = np.zeros(n+1) #RHS of eq
    y = np.zeros(n+1) 
    u = np.zeros(n+1) 


    A[0,0] = 2.0
    A[0,1] = -1.0
    for i in range(1, n): #define our matrix
        A[i,i-1] = -1.0
        A[i,i] = 2.0
        A[i,i+1] = -1.0
        ftilde[i] = hh*f(xi[i])  #to calculate RHS 


    A[n,n-1] = -1.0 
    A[n,n] = 2.0
    
    P, L, U = scipy.linalg.lu(A)   #built in function that gives us the the matrices L and U
    
    #below follows algorithm we found explained in the report and taken from the compendium by M. Hjorth-Jensen
    
    y[0] = ftilde[0]

    #forward sub. of L gives y
    for i in range(1, n):
        y[i] = ftilde[i]-y[i-1]*L[i,i-1]

    #backward sub. to get u
    u[n-1] = y[n-1]
    for i in range(n-2, 0, -1):
        u[i] = y[i] - U[i,i+1]*u[i+1]/U[i,i]

    return xi, u

def plot(N): 
    xi,u = lu_dec(N)
    plt.plot(xi,u, '--', label = 'n=%d' %N)
    plt.ylabel('Solution')
    plt.xlabel('x')
    plt.title('Analytical and numerical solution, LU decomposition')
    plt.legend(loc='upper right', frameon=False)
    plt.legend(loc='upper right', frameon=False)
 
def plot_exact(N):
    xi,u = lu_dec(N)
    exac = exact(N)
    plt.plot(xi, exac, label= 'Analytical'  )
    plt.legend(loc='upper right', frameon=False)   
    
plot(10**1)
plot(10**2)
plot(10**3)
plot(10**4)
plot_exact(10**4)



print("--- %s seconds ---" % (time.time() - start_time))

N = 10**1    #insert value up to 10**4 (above that doesnt work for LU decomp)
xi,u = lu_dec(N)
exac = exact(N)

max_err = []
abserr = np.abs((u[1:-1] - exac[1:-1])/exac[1:-1])
abserr = np.log10(abserr) # the first element is always 0
max_err.append(max(abserr))

print("n = %i    epsilon = %f\n" % (N, max(abserr)))








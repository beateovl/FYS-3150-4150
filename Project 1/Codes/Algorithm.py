#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:08:37 2020

@author: selmabeateovland
"""


import numpy as np
import matplotlib.pyplot as plt

import time
start_time = time.time() #to calculate the time in seconds for the program to run

class general: 
    def __init__(self, a, b, c, n):
        self.a = a #row under the diagonal
        self.b = b #diagonal
        self.c = c #row over the diagonal
        self.n = n 
        
        
        
    def f(self, x):
        return 100.0*np.exp(-10.0*x)   #calculate solution for f
    
    def anal(self, x): 
        return 1.0-(1.0-np.exp(-10.0))*x-np.exp(-10.0*x) #exact solution for our Poisson eq. 
    
    def x(self):     # x to be used in analytical sol., ftilde() and f(x)
        x = np.zeros(self.n+1)   #n+1 because want endpoints for the analytical sol, and len(exact)=len(u)
        h = 1.0/(self.n+1) #step size
        self.hh = h*h #faster
        for i in range(0,self.n+1):  
            x[i] = i*h
        return x
    
    def ftilde(self): #right hand side of our matrix vector problem
        x = self.x()
        ftilde = np.zeros(self.n+1)
        for i in range(0,self.n+1): 
            ftilde[i] = self.hh*self.f(x[i])
        return ftilde
    
    def exact(self,x): 
        x = self.x()
        exact = np.zeros(self.n+1)
        for i in range(0,self.n+1):
            exact[i] = self.anal(x[i])
        return exact
            
    
    
    def calc(self): #the general tridiagonal matrix function
        
            
        bstar = np.zeros(self.n)
        fstar = np.zeros(self.n)
        u = np.zeros(self.n+1) #include endpoints
        ab = np.zeros(self.n)
        ftilde = self.ftilde() #call function
        
        a = np.full(self.n-1, self.a) #to make an array full of elements with value a, size n-1 because the diagonal is size n
        c = np.full(self.n-1, self.c)
        b = np.full(self.n,self.b) #has one more element than a and b
        
        bstar[0] = b[0]
        fstar[0] = ftilde[0]
        u[0] = 0.0       #dirichlet bounday conditions
        u[self.n] = 0.0  #dirichlet bounday conditions
        
        #forward sub
        for i in range(1,self.n): 
            ab = a[i-1]/bstar[i-1]  # can be found in the calculation of both bstar and fstar, by calculating it here we save a flop
            bstar[i] = b[i] - c[i-1] * ab
            fstar[i] = ftilde[i] - fstar[i-1] * ab
        
        u[self.n-1] = fstar[self.n-1]/bstar[self.n-1] #to calculate u[i+1] in the next step
            
        #backward sub
        for i in range(self.n-2, 0, -1):
            u[i] = (fstar[i] - u[i+1] * c[i])/bstar[i] #our solution
        return u
    
    def special(self):      #our function when we input values for a, b and c (specialised alg)
        d = np.zeros(self.n+1)   #new 
        fstar = np.zeros(self.n)
        u = np.zeros(self.n+1)
        ftilde = self.ftilde()
        

        
        for i in range(1,self.n+1): 
            d[i-1] = (i + 1)/i   #i-1 because we are in python, and indexing starts at 0. Our d[0] = 2 is calculated here
                
        fstar[0] = ftilde[0]
        
        u[0] = 0.0       #dirichlet
        u[self.n] = 0.0  #dirichlet
            
        #forward sub
        for i in range(1,self.n): 
                fstar[i] = ftilde[i] + (fstar[i-1]/d[i-1])
                    
        u[self.n-1] = fstar[self.n-1]/d[self.n-1]
        
        #backward sub
        for i in range(self.n-2, 0, -1):
                u[i] = (fstar[i] + u[i+1])/d[i]
        return u

        
        
    def plot(self):           #to plot the general tridiagonal matrix alg
        cal = self.calc()   #call func, we need u 
        x = self.x()    #call func, we need x
        plt.plot(x, cal,'--', label = 'n=%d' %self.n)
        plt.ylabel('Solution')
        plt.xlabel('x')
        plt.title('A general tridiagonal matrix')
        plt.legend(loc='upper right', frameon=False)
        
    def plot_special(self):    #to plot the specialised case
        cal = self.special()  #call func, we need u
        x = self.x()      #call func, we need x
        plt.plot(x, cal,'--', label = 'n=%d' %self.n)
        plt.ylabel('Solution')
        plt.xlabel('x')
        plt.title('A tridiagonal matrix with identical elements along a, b and c')
        plt.legend(loc='upper right', frameon=False)
    
    def plot_exact(self):     #to plot the analytical solution
        x = self.x()
        exact = self.exact(x)
        plt.plot(x, exact, label= 'Analytical' )
        plt.legend(loc='upper right', frameon=False)
        
        


        
        

N = int(input('n='))
a = int(input('a='))
b = int(input('b='))
c = int(input('c='))

test = general(a,b,c,N)   #easier way to get plots of several values of n in one figure? Tried linspace, list, for loop, for n, but got an error message
test.plot_special()       #have to repeat the two first steps for each N you want in one plot
test.plot_exact()
plt.show()



#test.plot_exact() #testing if the analytical solution is the same for all n
#test2.plot_exact()  #seems like only n = 10 is slightly different
#test3.plot_exact()
#test4.plot_exact()
#test5.plot_exact()
#plt.show()

print("--- %s seconds ---" % (time.time() - start_time)) # I didn't manage to make a loop for this,
# to make the code run n times and find the average of it. Timeit didn't work. Had to do it manually 
#to find the average time it took to run the code for N = .... 


x = test.x()
u = test.calc()
exact = test.exact(x)
#
#
#
max_err = []
abserr = np.abs((u[1:-1] - exact[1:-1])/exact[1:-1]) #the first element is 0, we don't want to include that
abserr = np.log10(abserr) 
max_err.append(max(abserr))
print("n = %i    epsilon = %f\n" % (N, max(abserr)))




        
    
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm 

# Rosenbrock Funktion 

def f(x0,x1):
    return 100 * (x0**2 -x1)**2 + (x0-1)**2

def f_prime_x0(x0,x1):
    return 2*(200*x0*(x0**2 -x1) +x0 -1)

def f_prime_x1(x0,x1):
    return -200 * (x0**2-x1)


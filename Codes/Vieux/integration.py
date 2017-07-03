import matplotlib.pyplot as plt
import numpy as np

#Looking for a and b for y = a*x^b
def calculate_ab(xi, xf, yi, yf):
    logxi = np.log(xi)
    logxf = np.log(xf)
    logyi = np.log(yi)
    logyf = np.log(yf)
    b = (logyf - logyi)/(logxf - logxi)
    loga = logyi - b*logxi
    a = np.exp(loga)
    return a, b

#Calculate deltaS from deltaS = int from xi to xf a*x^b
#Calculate deltaS from deltaS = int from xi to xf a*x^b
def delta_S(xi, xf, yi, yf):
    [a, b] = calculate_ab(xi, xf, yi, yf)
    print(a, b)
    b = np.nan_to_num(b)
    deltaS = a/(b+1)*(xf**(b+1) - xi**(b+1))
    return deltaS

#Calculate total integral from init to final a*x^b
x = np.linspace(0.5, 10, 100)
y = 3*x**2
deltaS = 0

integral = 0

for i in range (1, len(x)):
    deltaS = delta_S(x[i-1], x[i], y[i-1], y[i])
    integral = integral + deltaS

print(integral)
plt.xscale('log')
plt.yscale('log')
plt.plot(x, y)
plt.show()

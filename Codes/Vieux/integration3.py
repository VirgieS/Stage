import matplotlib.pyplot as plt
import numpy as np

def integration_semilog(x, y):

    #Looking for a and b for y = a*x^b
    def calculate_ab(xi, xf, yi, yf):
        logxi = np.log(xi)
        logxf = np.log(xf)
        a = (yf - yi)/(logxf - logxi)
        b = yi - a*logxi
        print(a, b)
        return a, b

    #Calculate deltaS from deltaS = int from xi to xf a*x^b
    def delta_S(xi, xf, yi, yf):
        [a, b] = calculate_ab(xi, xf, yi, yf)
        return a * (xf * np.log(xf) - xi * np.log(xi) - (xf - xi)) + b *(xf - xi)

    deltaS = np.zeros(len(x)-1)
    integral = 0

    for i in range (1, len(x)):
        deltaS[i-1] = delta_S(x[i-1], x[i], y[i-1], y[i])
        integral = integral + deltaS[i-1]

    return integral

#Calculate total integral from init to final a*x^b
x = np.linspace(0.5, 10000, 10000000)
y = 3.0

integral = integration_semilog(x, y)
print(integral)

plt.xscale('log')
plt.plot(x, y)
plt.show()

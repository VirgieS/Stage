import matplotlib.pyplot as plt
import numpy as np

from principal_functions import integration

x=np.linspace(0.5, 1, 100)
y = 5*x**7

integral = integration(x, y)

print(integral)

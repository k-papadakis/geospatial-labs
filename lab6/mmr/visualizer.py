import numpy as np
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
plt.plot(x, x**2)
plt.show()
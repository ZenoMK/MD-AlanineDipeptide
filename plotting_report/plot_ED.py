import matplotlib.pyplot as plt
import numpy as np

X = np.random.rand(2,45)

X = X[15:]

plt.plot(X)
plt.show()
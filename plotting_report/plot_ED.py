import matplotlib.pyplot as plt
import numpy as np

X = [[0,330.0,0.05],
[1,290.0,0.25],
[2,310.0,0.1],
[3,280.0,0.25],
[4,300.0,0.1],
[5,370.0,0.15],
[6,310.0,0.2],
[7,320.0,0.2],
[8,350.0,0.1],
[9,290.0,0.05],
[10,340.0,0.15],
[11,350.0,0.2],
[12,370.0,0.15],
[13,340.0,0.1],
[14,320.0,0.2],
[15,370.0,0.25],
[16,280.0,0.05],
[17,370.0,0.05],
[18,280.0,0.25],
[19,370.0,0.25],
[20,280.0,0.05],
[21,370.0,0.05]]

X = X[15:]

plt.plot(X)
plt.show()
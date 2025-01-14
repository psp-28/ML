import numpy as np
import matplotlib.pyplot as plt

y = np.array([98,80,68,10,20,56,68,45])
x = np.array([10,9,8,2,6,8,4,9])

for i in range(0,len(x)):
    plt.plot(x[i],y[i],"gX")


m, c = np.polyfit(x, y, 1)
y = m*x + c

for i in range(0,len(x)):
    plt.plot(x[i],y[i], "bo")

plt.plot(x, y, '-r')
plt.title("Linear Regression")
plt.xlabel("No. of hours spend driving")
plt.ylabel("Risk factor")

plt.show()
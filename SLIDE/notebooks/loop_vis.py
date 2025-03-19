import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir("/home/weidemaier/PDE Net")
print(os.getcwd())
error_vec = [None]*2
with open("overview.txt", "r") as filestream:
    i = 0
    for line in filestream:
        currentline = line.split(",")
        error_vec[i] = np.float32(currentline[1]) #, np.float32(currentline[2]), np.float32(currentline[3][1])
        L2 = currentline[1]
        w12 = currentline[2]
        diff = currentline[3]
        i += 1
    

plt.style.use('_mpl-gallery')

# make data
np.random.seed(1)
x = 4 + np.random.normal(0, 1.5, 200)

# plot:
fig, ax = plt.subplots()
print(error_vec)
ax.bar([0,1], height = error_vec)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, np.max(error_vec)), yticks=np.linspace(0,np.max(error_vec), 10))
fig.savefig("tests.png", transparent=None)
plt.show()

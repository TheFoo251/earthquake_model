from matplotlib import pyplot as plt
import numpy as np
import csv
import sys

PATH = sys.argv[1]

epoch, loss, val_loss = [], [], []
with open(PATH) as file:
    reader = csv.reader(file, delimiter=',')
    lines = list(reader)[1:]
    epoch = np.array([int(line[0]) for line in lines])
    loss = np.array([float(line[1]) for line in lines])
    val_loss = np.array([float(line[2]) for line in lines])

fig, ax = plt.subplots()
ax.plot(epoch, loss, 'D--')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

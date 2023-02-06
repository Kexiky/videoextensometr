import numpy as np
import matplotlib.pyplot as plt

def vectors_displacement(img1, vec, n_step):
    plt.plot(img1)
    for i in np.range(n_step):
        plt.plot(vec[i, ])
    return 0

#Методы трекинка частей изображения

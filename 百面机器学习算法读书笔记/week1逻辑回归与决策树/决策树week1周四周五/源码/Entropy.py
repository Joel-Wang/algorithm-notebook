# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 21:32:55 2019

@author: Joel Wang
"""

import numpy as np
import matplotlib.pyplot as plt

def entropy(p):
    return -p*np.log(p)-(1-p)*np.log(1-p)

x = np.linspace(0.01,0.99,200)

plt.plot(x,entropy(x))
plt.show
import random as rand
import numpy as np
from noelle_model import model

size_space = [0, 1, 2, 3]
angle_space = [0, 1, 2, 3]

training_items = np.matrix([[3,0], #first item is size, second is angle
                            [2,1],
                            [2,2],
                            [1,1],
                            [1,3],
                            [0,0],
                            [0,2]])


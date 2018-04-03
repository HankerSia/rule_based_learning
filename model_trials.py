import random as rand
import numpy as np
from noelle_model import model

size_space = [0, 1, 2, 3]
angle_space = [0, 1, 2, 3]

#training items for all three cases
#first item is angle, second is size
training_items = np.array([[0,3], #M
                            [1,2], #J
                            [2,2], #K
                            [1,1], #F
                            [3,1], #H
                            [0,0], #A
                            [2,0]])#C

#[1,0] = black, [0,1] = white
training_targets = np.array([[1,0], #M
                             [0,1], #J
                             [0,1], #K
                             [0,1], #F
                             [1,0], #H
                             [1,0], #A
                             [1,0]])#C

def test(m, case_performance):
    trial_performance = []
    for size in size_space:
        init = 0
        for angle in angle_space:
            m.applyNoise(uninstructed.inst_to_category)
            m.customInput(size, angle, instructed=False, feedback=False)
            m.feedForward(False)
            m.resetNoise(uninstructed.inst_to_category)
            """
            print("Output on exemplar w/ size " + str(size) +" and angle " + str(angle) + ":")
            print(m.category)
            """
            if(m.category[0,0] > m.category[0,1]):
                exemplar_performance = [0,1]
            else:
                exemplar_performance = [1,0]
            trial_performance.append(exemplar_performance)
    trial_performance = np.array(trial_performance)
    case_performance += trial_performance
        

"""
UNINSTRUCTED CASE:  activations of the working memory layer are fixed to zero
to simulate the absence of a rule guideline. The network is trained only on the
training items/targets until it correctly identfies them and then
it is tested on all possible stimuli.
To reach acceptable classification rates on the examples in this case, a learning
rate of 1 and 21000 iterations were required.
The results are the compilation of 100 tests.
"""

uninstructed_performance = np.zeros((16, 2))
uninstructed = model()
uninstructed.importWeights()
uninstructed.learning_rate = 1

for i in range(21000): #choose a random training example
    ex = np.random.randint(7)
    uninstructed.customInput(training_items[ex,0], training_items[ex,1], training_targets[ex], instructed=False)
    uninstructed.feedForward(False)

    """
    if(i % 10000 == 0):
        print(i)
        print("targets: \n" + str(uninstructed.target))
         print("output: \n" + str(uninstructed.category))
        print("error: \n" + str(np.mean(np.square(uninstructed.target - uninstructed.category))))
        print("_______")
    """
    
    uninstructed.feedBackward(False)

for i in range(100): #test the networks performance on all 16 stimuli combinations, then compute average classification rates
    test(uninstructed, uninstructed_performance)
uninstructed_performance /= 100
print(uninstructed_performance)


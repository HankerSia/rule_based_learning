import random as rand
import numpy as np
from noelle_model import model

#all possible stimuli combinations
size_space = [0, 1, 2, 3]
angle_space = [0, 1, 2, 3]

#elements 0-3: size to be placed in black category
#elements 4-7: angle to be placed in black category
#elements 8-11: size to be placed in white category
#elements 12-15: angle to be placed in white category
#element 16 is bias
simple_training_rule = [1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1]
simple_rule = [1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1]
complex_training_rule = [1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1]
complex_rule = [1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1]

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


#used to convert output into probabilities
def softmax(inputs):
    return np.exp(inputs) / float(sum(np.exp(inputs)))

#compiles a given cases' performance on all 16 possible stimuli combinations
def test(m, case_performance, r=[], instructed=True):
    trial_performance = []
    for size in size_space:
        for angle in angle_space:
            if(instructed):
                m.customInput(size, angle, rule=r, instructed=True, feedback=False)
                m.feedForward(True, False)
            else:
                m.customInput(size, angle, instructed=False, feedback=False)
                m.feedForward(False, False)
            """
            print("Output on exemplar w/ size " + str(size) +" and angle " + str(angle) + ":")
            print(m.category)
            """
            exemplar_performance = [m.category[0,0], m.category[0, 1]] #decompose category activations into a list before appending them
            trial_performance.append(exemplar_performance)
    trial_performance = np.array(trial_performance)
    case_performance += trial_performance
        

"""
UNINSTRUCTED CASE: activations of the working memory layer are fixed to zero
to simulate the absence of rule following. only weights between the output and
the degraded stimulus layer are modified. The network is trained only on the
training items/targets until it correctly identfies them and then
it is tested on all possible stimuli.
To reach acceptable classification rates on the examples in this case, a learning
rate of 1 and 21000 iterations were required.
The results are the result of applying a softmax function to the sum total of
activations in the category layer for each exemplar.
"""

uninstructed_performance = np.zeros((16, 2))
uninstructed = model()
uninstructed.importWeights()
uninstructed.learning_rate = 1

for i in range(21000): #choose a random training example
    ex = np.random.randint(7)
    uninstructed.customInput(training_items[ex,0], training_items[ex,1], training_targets[ex], instructed=False)
    uninstructed.feedForward(False, False)

    """
    if(i % 10000 == 0):
        print(i)
        print("targets: \n" + str(uninstructed.target))
         print("output: \n" + str(uninstructed.category))
        print("error: \n" + str(np.mean(np.square(uninstructed.target - uninstructed.category))))
        print("_______")
    """
    
    uninstructed.feedBackward(False)

for i in range(50): #test the networks performance on all 16 stimuli combinations, then compute average classification rates
    test(uninstructed, uninstructed_performance, instructed=False)

for i in range(16):
    uninstructed_performance[i] = softmax(uninstructed_performance[i])
print(uninstructed_performance)


"""
SIMPLE RULE CASE: only weights between the output and
the degraded stimulus layer are modified. The network is trained on the
training items/targets and the training simple rule until it correctly identfies them.
It's then tested for its classification performance on all
possible stimuli given a slightly modified simple rule.
The network was trained on the 7 training exemplars and the simple training rule
for 252 iterations, with a learning rate of 0.5.
The results are the result of applying a softmax function to the sum total of
activations in the category layer for each exemplar.
"""

simple_performance = np.zeros((16, 2))
simple = model()
simple.importWeights()
simple.learning_rate = 0.5

test(simple, simple_performance, r=simple_training_rule, instructed=True)

for i in range(252): #choose a random training example
    ex = np.random.randint(7)
    simple.customInput(training_items[ex,0], training_items[ex,1], training_targets[ex], simple_training_rule)
    simple.feedForward(True, False)
    """
    if(i % 500 == 0):
        print(i)
        print("targets: \n" + str(simple.target))
        print("output: \n" + str(simple.category))
        print("error: \n" + str(np.mean(np.square(simple.target - simple.category))))
        print("_______")
    """   
    simple.feedBackward(False)

for i in range(50): #test the networks performance on all 16 stimuli combinations, then compute average classification rates
    test(simple, simple_performance, r=simple_rule, instructed=True)

for i in range(16):
    simple_performance[i] = softmax(simple_performance[i])
print(simple_performance)


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
simple_test_rule = [1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1]
simple_rule = [1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1]
complex_test_rule = [1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1]
complex_rule = [1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1]

#first item is size, second is angle
training_items =  np.array([[3,0], #M
                            [2,1], #J
                            [2,2], #K
                            [1,1], #F
                            [1,3], #H
                            [0,0], #A
                            [0,2]])#C

#[1,0] = black, [0,1] = white
training_targets = np.array([[1,0], #M
                             [0,1], #J
                             [0,1], #K
                             [0,1], #F
                             [1,0], #H
                             [1,0], #A
                             [1,0]])#C


#converts output into probabilities
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
uninstructed.applyNoise(uninstructed.inst_to_category)
uninstructed.applyNoise(uninstructed.inst_to_category)
uninstructed.applyNoise(uninstructed.inst_to_category)
uninstructed.applyNoise(uninstructed.inst_to_category)
uninstructed.learning_rate = 1.

for i in range(30000): #trains the uninstructed network
    ex = np.random.randint(7)
    uninstructed.customInput(training_items[ex,0], training_items[ex,1], training_targets[ex], instructed=False)
    uninstructed.feedForward(False, False)
    uninstructed.feedBackward(False)

    """
    if(ex == 4):
        print(i)
        print("size/angle: " + str(training_items[ex]))
        print("targets: \n" + str(uninstructed.target))
        print("output: \n" + str(uninstructed.category))
        print("error: \n" + str(np.mean(np.square(uninstructed.target - uninstructed.category))))
        print("_______")
     """
    

"""
SIMPLE RULE CASE: only weights between the output and
the degraded stimulus layer are modified. The network is initially tested on the
testing items/targets and its performance on all 16 stimuli combinations given
a simple rule are recorded. It's then trained on the training examples and targets,
as well as a slightly modified simple rule. The network was trained on the 7
training exemplars and the simple rule for 252 iterations, with a learning rate of 0.5.
The results are the result of applying a softmax function to the sum total of
activations in the category layer for each exemplar.

COMPLEX RULE CASE: identical to the simple rule case, but with a rule with a negative element.
"""

simple_test_performance = np.zeros((16, 2))
simple_performance = np.zeros((16,2))
simple = model()
simple.importWeights()
simple.learning_rate = 0.5

complex_test_performance = np.zeros((16, 2))
complex_performance = np.zeros((16, 2))
complx = model()
complx.importWeights()
complx.learning_rate = 0.5

for i in range(50): #compile the networks' performance on simple/complex rules with no training
    test(simple, simple_test_performance, r=simple_test_rule, instructed=True)
    test(complx, complex_test_performance, r=complex_test_rule, instructed=True)
    
for i in range(252): #trains the simple and complex rule-following networks
    ex = np.random.randint(7)
    simple.customInput(training_items[ex,0], training_items[ex,1], training_targets[ex], simple_rule)
    simple.feedForward(True, False)
    simple.feedBackward(False)

    ex = np.random.randint(7)
    complx.customInput(training_items[ex,0], training_items[ex,1], training_targets[ex], complex_rule)
    complx.feedForward(True, False)
    complx.feedBackward(False)

for i in range(50): #compile the networks' performance on modified simple/complex rules after training
    test(uninstructed, uninstructed_performance, instructed=False)
    test(simple, simple_performance, r=simple_rule, instructed=True)
    test(complx, complex_performance, r=complex_rule, instructed=True)

for i in range(16): #compute average classification rates by applying a softmax function to the sum total activation rates
    uninstructed_performance[i] = softmax(uninstructed_performance[i])
    simple_performance[i] = softmax(simple_performance[i])
    simple_test_performance[i] = softmax(simple_test_performance[i])
    complex_performance[i] = softmax(complex_performance[i])
    complex_test_performance[i] = softmax(complex_test_performance[i])

print('uninstructed results: ')
print(uninstructed_performance)
print("----")
print('simple rule testing results: ')
print(simple_test_performance)
print('simple rule following results: ')
print(simple_performance)
print("----")
print('complex rule testing results: ')
print(complex_test_performance)
print('complex rule following results: ')
print(complex_performance)


import numpy as np

class model:
    def __init__(self):
        self.learning_rate = 0.05
        
        #likelihood of 0's in rule categories
        self.pos_rule_dist = 0.2
        self.neg_rule_dist = 0.8

        #bias value used to generate sparse representations
        self.sparsity_bias = -3.0

        #inital randomized synapses, exclude layers with static weights
        self.degraded_to_hidden = (1*np.random.random((9,25))-0.5) #bounded by [-0.5, 0.5)
        self.rules_to_working = (0.4*np.random.random((17,25))) #bounded by [0, 0.4)
        self.inst_to_category = (1*np.random.random((17, 2))-0.5) #bounded by [-0.5, 0,5)
        self.working_to_hidden = (2*np.random.random((25,25))-1) #no bounds, [-1.0, 1.0)
        self.hidden_to_category = (1*np.random.random((25,2))-0.5) #bounded by [-0.5, 0.5)

        
        #lists containing activations for each layer
        self.stim_size = 0
        self.stim_angle = 0
        self.rules = []
        self.degraded_stim = []
        self.working = []
        self.instances = []
        self.hidden = []
        self.category = [] #element 0 represents black, and element 1 represents white
        self.target = [0]*2 #target category activations

    #activation function & derivative for delta calcuation
    def sigmoid(self, x, deriv=False):
        if(deriv == True):
            return (x*(1-x))
        return 1/(1+np.exp(-x))

    #used to convert output into probabilities
    def softmax(self, inputs):
        return np.exp(inputs) / float(sum(np.exp(inputs)))

    """
    single rows are used to represent 2-dimensional data sets for ease of computation.

    instances: 16 elements corresponding to all possible combinations of stim_size and stim_angle.
    each block of 4 elements corresponds to the addition of each element of stim_angle to one element of stim_size.
    ---
    rules: 16 elements corresponding to all positive and negative rule combinations. The first 8 elements are the positive rule,
    with the first four being size and the second being angle. The second set of 8 elements are the negative rule following the same conventions.
    """

    #generate random activations for the three stimuli layers
    def genStim(self): 
        self.stim_size = np.random.randint(4)
        self.stim_angle = np.random.randint(4)

        d_stim = []
        if(self.stim_size == 0):
            d_stim += [1.0, .449, .202, .091]
        if(self.stim_size == 1):
            d_stim += [.449, 1.0, .449, .202]
        if(self.stim_size == 2):
            d_stim += [.202, .449, 1.0, .449]
        if(self.stim_size == 3):
            d_stim += [.091, .202, .449, 1.0]

        if(self.stim_angle == 0):
            d_stim += [1.0, .449, .202, .091]
        if(self.stim_angle == 1):
            d_stim += [.449, 1.0, .449, .202]
        if(self.stim_angle == 2):
            d_stim += [.202, .449, 1.0, .449]
        if(self.stim_angle == 3):
            d_stim += [.091, .202, .449, 1.0]

        d_stim.append(self.sparsity_bias)
        self.degraded_stim = np.array(d_stim).reshape(1, -1)

        inst = []
        for i in range(4):
            for j in range(4, 8):
              inst.append(self.sigmoid(self.degraded_stim[0][i] * self.degraded_stim[0][j]))
        inst.append(self.sparsity_bias)
        self.instances = np.array(inst).reshape(1, -1)

    #generate random rule combinations
    def genRules(self, complex_rule): #generate random rule combinations
        r = [0]*17
        for j in range(8):#positive rule inputs
            x = np.random.random()
            if(x >= self.pos_rule_dist):
                r[j] = 1
        if(complex_rule):#negative rule inputs stay zero if simple rule is called for
            for k in range(8, 16):
                x = np.random.random()
                if(x >= self.neg_rule_dist):
                    r[k] = 1
        r[16] = 1 #bias unit
        self.rules = np.array(r).reshape(1, -1)

    #determines the target output from given rules & stimuli
    def genTarget(self, complex_rule):
        #check for discrepancies between stimulus and positive rule, if any are found then the target is white (otherwise it's black)
        if(self.rules[0][self.stim_size] != 1):
            return [0, 1]
        if(self.rules[0][self.stim_angle + 4] != 1):
            return [0, 1]
        #check for discrepancies between stimulus and negative rule if there is one
        if(complex_rule):
            if(self.rules[0][self.stim_size + 8] == 1):
                return [0, 1]
            if(self.rules[0][self.stim_size + 12] == 1):
                return [0, 1]
        return [1, 0]
        
    #populates the input set consisting of rules, stimuli, and the correct output for 1 trial
    def genInput(self, complex_rule=True): 
        self.genStim()
        self.genRules(complex_rule)
        self.target = np.array(self.genTarget(complex_rule)).reshape(1, -1)
        
    #calculate activations for working memory, hidden and category layers
    def feedForward(self, instructed=True):

        if(instructed):
            self.working = self.sigmoid(np.dot(self.rules, self.rules_to_working)).reshape(1, -1)
            self.working[0][24] = self.sparsity_bias
        else:
            self.working = np.array(0*[25]).reshape(1, -1)

        self.hidden = self.sigmoid(np.dot(self.working, self.working_to_hidden) + np.dot(self.degraded_stim, self.degraded_to_hidden)).reshape(1, -1)
        self.hidden[0][24] = self.sparsity_bias
        self.category = self.sigmoid(np.dot(self.hidden, self.hidden_to_category) + np.dot(self.instances, self.inst_to_category))
        self.category = self.softmax(self.category[0]).reshape(1, -1)

        """if(prnt == 1):
            print("instance activations: " + str(self.instances))
            print("instance weights: " + str(self.inst_to_category))
            print("result of dot: " + str(self.sigmoid(np.dot(self.instances, self.inst_to_category))))
            print(self.category) """

    #calculate and apply weight changes for all synapse groups
    def feedBackward(self):
    
        out_error = self.target - self.category
        out_delta = out_error*self.sigmoid(self.category, True)

        inst_error = out_delta.dot(self.inst_to_category.T)
        inst_delta = inst_error*self.sigmoid(self.instances, True)

        hid_error = out_delta.dot(self.hidden_to_category.T)
        hid_delta = hid_error*self.sigmoid(self.hidden, True)

        dstim_error = hid_delta.dot(self.degraded_to_hidden.T)

        working_error = hid_delta.dot(self.working_to_hidden.T)
        working_delta = working_error*self.sigmoid(self.working, True)

        rule_error = working_delta.dot(self.rules_to_working.T)
        
        self.inst_to_category += self.learning_rate*self.instances.T.dot(out_delta)
        self.hidden_to_category += self.learning_rate*self.hidden.T.dot(out_delta)
        self.degraded_to_hidden += self.learning_rate*self.degraded_stim.T.dot(hid_delta)
        self.working_to_hidden += self.learning_rate*self.working.T.dot(hid_delta)
        self.rules_to_working += self.learning_rate*self.rules.T.dot(working_delta)

    #inital training of the network
    def train(self): 
        for i in range(5000000):
            self.genInput()
            self.feedForward()
            self.feedBackward()
            if(i % 100000 == 0):
                print("rules: " + str(self.rules) + ", square was size " + str(self.stim_size) + " and angle " + str(self.stim_angle))
                print("resulting target: " + str(self.target))
                print("output: " + str(self.category))
                print("error: " + str(np.mean(np.square(self.target - self.category))))
                print("_______")
            
    #exports each weight matrix to 5 files
    def export(self): 
        f = open('degraded_to_hidden.txt', 'w')
        for syn in self.degraded_to_hidden:
            f.write("%s\n" % syn)
        f.close()

        f = open('rules_to_working.txt', 'w')
        for syn in self.rules_to_working:
            f.write("%s\n" % syn)
        f.close()

        f = open('inst_to_category.txt', 'w')
        for syn in self.inst_to_category:
            f.write("%s\n" % syn)
        f.close()

        f = open('working_to_hidden.txt', 'w')
        for syn in self.working_to_hidden:
            f.write("%s\n" % syn)
        f.close()

        f = open('hidden_to_category.txt', 'w')
        for syn in self.hidden_to_category:
            f.write("%s\n" % syn)
        f.close()

    #tests the network on 10 random inputs
    def test(self): 
        for i in range(11):
            self.genInput()
            self.feedForward()
            print("rules: " + str(self.rules) + ", square was size " + str(self.stim_size) + " and angle " + str(self.stim_angle))
            print("resulting target: " + str(self.target))
            print("output: " + str(self.category))
            print("error: " + str(np.mean(np.square(self.target - self.category))))
            print("_______")
    



            

      

        
        




       


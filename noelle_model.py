import numpy as np

class model:
    def __init__(self):
        
        #affects complexity of rules, higher value = higher chance of complexity
        self.pos_rule_dist = 0.5
        self.neg_rule_dist = 0.5

        #used for debugging
        self.trial_type = 0

        #bias value used to generate sparse representations
        self.sparsity_bias = -3.0

        #learning rate
        self.learning_rate = 0.005

        #noise level for trial cases (the standard deviation used for random sampling of the normal dist.)
        #bigger = more noise
        self.noise = 0.3
        self.curr_noise = 0.0 #store the generated noise value so that it can be removed

        #inital randomized synapses, exclude layers with static weights
        self.degraded_to_hidden = (1*np.random.random((9,25))-0.5) #bounded by [-0.5, 0.5)
        self.rules_to_working = (0.4*np.random.random((17,25))) #bounded by [0, 0.4)
        self.inst_to_category = (1*np.random.random((17, 2))-0.5) #bounded by [-0.5, 0,5)
        self.working_to_hidden = (2*np.random.random((25,25))-1) #no bounds, [-1.0, 1.0)
        self.hidden_to_category = (1*np.random.random((25,2))-0.5) #bounded by [-0.5, 0.5)
        
        #lists containing activations for each layer
        self.stim_size = self.stim_angle = 0
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

    #imports weight matricies from text files
    def importWeights(self):
        self.degraded_to_hidden = np.loadtxt('degraded_to_hidden.txt')
        self.rules_to_working = np.loadtxt('rules_to_working.txt')
        self.inst_to_category = np.loadtxt('inst_to_category.txt')
        self.working_to_hidden = np.loadtxt('working_to_hidden.txt')
        self.hidden_to_category = np.loadtxt('hidden_to_category.txt')

    """
    NOTE: the weights currently stored in the text files were generated with the following configurations:
    pos_rule_dist = 0.5
    neg_rule_dist = 0.5
    learning_rate = 0.005
    training iterations: 8000000
    error rate: 0.048
    """

    #exports each weight matrix to 5 files
    def exportWeights(self): 
        f = open('degraded_to_hidden.txt', 'w')
        for syn in self.degraded_to_hidden:
            for val in syn:
                f.write("%s " % val)
            f.write("\n")
        f.close()

        f = open('rules_to_working.txt', 'w')
        for syn in self.rules_to_working:
            for val in syn:
                f.write("%s " % val)
            f.write("\n")
        f.close()

        f = open('inst_to_category.txt', 'w')
        for syn in self.inst_to_category:
            for val in syn:
                f.write("%s " % val)
            f.write("\n")
        f.close()

        f = open('working_to_hidden.txt', 'w')
        for syn in self.working_to_hidden:
            for val in syn:
                f.write("%s " % val)
            f.write("\n")
        f.close()

        f = open('hidden_to_category.txt', 'w')
        for syn in self.hidden_to_category:
            for val in syn:
                f.write("%s " % val)
            f.write("\n")
        f.close()

    """
    single rows are used to represent 2-dimensional data sets for ease of computation.

    instances: 16 elements corresponding to all possible combinations of stim_size and stim_angle.
    each block of 4 elements corresponds to the multiplication of each element of stim_angle to one element of stim_size.
    ---
    rules: 16 elements corresponding to all positive and negative rule combinations. The first 8 elements are the positive rule,
    with the first four being size and the second four being angle. The second set of 8 elements are the negative rule following the same conventions.
    """

    #generate activations for the three stimuli layers. must be called before every call to genRules/Target
    #can generate specific stimuli combinations when given corresponding arguments
    def genStim(self, size=-1, angle=-1):
        if(size == -1 and angle == -1):
            self.stim_size = np.random.randint(4)
            self.stim_angle = np.random.randint(4)
        else:
            self.stim_size = size
            self.stim_angle = angle

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
        
        inst = []
        for i in range(4):
            for j in range(4, 8):
              inst.append(self.sigmoid(d_stim[i] * d_stim[j]))
        inst.append(self.sparsity_bias)
        #returns activations for both degraded stimulus and instances layers
        return (np.array(d_stim), np.array(inst))

    #generate random rule combinations
    #when color is 0, rules are created such that the target will be black.
    #1 is for white, and 2 is for white due to complex rules 
    #only call this after generating stimuli
    def genRules(self, color=-1): 
        r = [0]*17
        overwrite = 0 #for color == 2. stores which black rule was assigned
        if(color == 0 or color == 2): #ensure that at least one matching rule is chosen to satisfy black condition
            if (np.random.random() >= 0.5):
                r[self.stim_size] = 1
                overwrite = self.stim_size+8
            else:
                r[self.stim_angle+4] = 1
                overwrite = self.stim_angle+12
        #random positive and negative rule generation
        for j in range(8):              
            if(np.random.random() <= self.pos_rule_dist):
                r[j] = 1
            if(np.random.random() <= self.neg_rule_dist):
                r[j + 8] = 1
        if(color == 0): #nullify conflicting negative rules for black
            r[self.stim_size+8] = 0
            r[self.stim_angle+12] = 0
        elif(color == 1): #nullify conflicting positive rules for white
            r[self.stim_size] = 0
            r[self.stim_angle+4] = 0
        elif(color == 2): #if we want to create a scenario where a complex rule
        #overrides a black rule, choose corresponding complex rules after selecting black's
            r[overwrite] = 1
            
        r[16] = 1 #bias unit
        return np.array(r)

    #determines the target output from given rules & stimuli
    #can be given an int representing the target if the target category is known (0 for black, 1 for white)
    def genTarget(self, given=-1):
        if(given == 0):
            return np.array([1, 0])
        if(given == 1 or given == 2):
            return np.array([0, 1])
        
        #check for discrepancies between stimulus and positive rule, if neither corresponding rule is active then it's white
        if(self.rules[0][self.stim_size] != 1 and self.rules[0][self.stim_angle + 4] != 1):
            return np.array([0, 1])
        #check for discrepancies between stimulus and negative rule, if either corresponding rule is active then it's white
        if(self.rules[0][self.stim_size + 8] == 1 or self.rules[0][self.stim_angle + 12] == 1 ):
            return np.array([0, 1])
        #black stimuli pass all the above criteria
        return np.array([1, 0])

    #populates the input activations with either category of training example 
    #or completely randomized examples 
    def genInput(self, categoried=True):
        stim = self.genStim()
        self.degraded_stim = stim[0].reshape(1, -1)
        self.instances = stim[1].reshape(1, -1)
        if(categoried): #randomly generate a specific type of training examplar
            #trial type = 0: black
            #trial type = 1: white by simple or complex rule
            #trial tpe = 2: random
            self.trial_type = np.random.randint(3)
            if(self.trial_type == 2):
                self.rules = self.genRules().reshape(1, -1)
                self.target = self.genTarget().reshape(1, -1)
            elif(self.trial_type == 1):
                self.trial_type = np.random.randint(1, 3)
                self.rules = self.genRules(self.trial_type).reshape(1, -1)
                self.target = self.genTarget(self.trial_type).reshape(1, -1)
            else:
                self.rules = self.genRules(0).reshape(1, -1)
                self.target = self.genTarget(0).reshape(1, -1)

        else: #pure random rule generation
            self.trial_type = 3
            self.rules = self.genRules().reshape(1, -1)
            self.target = self.genTarget().reshape(1, -1)

    #input custom rule/stimuli combinations
    def customInput(self, size, angle, target = [], rule = [], instructed=True, feedback=True):
        stim = self.genStim(size, angle)
        self.degraded_stim = stim[0].reshape(1, -1)
        self.instances = stim[1].reshape(1,-1)
        if(instructed): #only populate rules in instructed cases
            self.rules = rule.reshape(1, -1)
        if(feedback): #only populate targets for experimental training
            self.target = target.reshape(1,-1)

    #calculate activations for working memory, hidden and category layers
    #when uninstructed, fix working memory activations to zero
    def feedForward(self, instructed=True):
        if(instructed):
            self.working = self.sigmoid(np.dot(self.rules, self.rules_to_working)).reshape(1, -1)
            self.working[0][24] = self.sparsity_bias
            self.hidden = self.sigmoid(np.dot(self.working, self.working_to_hidden) + np.dot(self.degraded_stim, self.degraded_to_hidden))
        else:
            self.hidden = self.sigmoid(np.dot(self.degraded_stim, self.degraded_to_hidden))

        self.hidden[0][24] = self.sparsity_bias
        self.category = self.sigmoid(np.dot(self.hidden, self.hidden_to_category) + np.dot(self.instances, self.inst_to_category))

    #calculate and apply weight changes for all synapse groups
    def feedBackward(self, initialization=True):
        out_error = self.target - self.category
        out_delta = out_error*self.sigmoid(self.category, True)
        
        if(initialization): #only modify hidden layer, degraded stim, and working memory weights during initial training
            hid_error = out_delta.dot(self.hidden_to_category.T)
            hid_delta = hid_error*self.sigmoid(self.hidden, True)

            dstim_error = hid_delta.dot(self.degraded_to_hidden.T)

            working_error = hid_delta.dot(self.working_to_hidden.T)
            working_delta = working_error*self.sigmoid(self.working, True)

            rule_error = working_delta.dot(self.rules_to_working.T)
            
        self.inst_to_category += self.learning_rate*self.instances.T.dot(out_delta)

        if(initialization):
            self.hidden_to_category += self.learning_rate*self.hidden.T.dot(out_delta)
            self.degraded_to_hidden += self.learning_rate*self.degraded_stim.T.dot(hid_delta)
            self.working_to_hidden += self.learning_rate*self.working.T.dot(hid_delta)
            self.rules_to_working += self.learning_rate*self.rules.T.dot(working_delta)

    #train the network on generated inputs
    def train(self, iterations= 8000000): 
        for i in range(iterations):
            self.genInput()
            self.feedForward()
            self.feedBackward()
            #print body for debugging during initial training
            """
            if(i % (iterations*0.01) == 0):
                print("progress: " + str(i) + "/10000000")
                print("rules: \n" + str(self.rules) + "\nsquare was size " + str(self.stim_size) + " and angle " + str(self.stim_angle))
                if(self.trial_type == 0):
                    print("black condition")
                elif(self.trial_type == 1):
                    print("white condition")
                elif(self.trial_type == 2):
                    print("complex condition")
                else:
                    print("random condition")
                print("targets: \n" + str(self.target))
                print("output: \n" + str(self.category))
                print("error: \n" + str(np.mean(np.square(self.target - self.category))))
                print("_______")
            """        
    
    #tests the network on random or categoried inputs (for debugging)
    def test(self, iterations=1000, categoried=False, display=False): 
        if(display):
            for i in range(11): #small amount of exemplars with more info displayed per trial
                self.genInput(categoried)
                self.feedForward()
                print("rules: " + str(self.rules) + ", \nsquare was size " + str(self.stim_size) + " and angle " + str(self.stim_angle))
                if(self.trial_type == 0):
                        print("black condition")
                elif(self.trial_type == 1):
                    print("white condition")
                elif(self.trial_type == 2):
                    print("complex condition")
                else:
                    print("random condition")
                print("targets: " + str(self.target))
                print("output: " + str(self.category))
                print("error: " + str(np.mean(np.square(self.target - self.category))))
                print("_______")
        else:
            sse_avg = 0
            err_count = 0
            for i in range(0, iterations): #calculate average error over larger number of trials
                self.genInput(categoried)
                self.feedForward()
                err = np.mean(np.square(self.target - self.category))
                if(err >= 0.25):
                    err_count += 1

                sse_avg += np.mean(np.square(self.target - self.category))
    
            print("average error: " + str(sse_avg / iterations))
            print("mis-classifications: " + str(err_count) + "/" + str(iterations))

    def applyNoise(self, matrix):
        noise = np.random.normal(0, self.noise, matrix.shape)
        self.curr_noise = noise
        matrix -= np.abs(noise)

    def resetNoise(self, matrix):
        matrix += np.abs(self.curr_noise)
        self.curr_noise = 0

            
    



            

      

        
        




       


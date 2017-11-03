###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids:
# Ankit Saxena (ansaxena)
# Bhargavi Chalasani (bchalasa)
# Parag Juneja (pjuneja)
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
#In the train method, we have calculated the probabilites of all words as the first word P(S_1) in the sentence (in dictionary: word_initial)
#The word emission probabilities P(W_i, S_i) are calculated (in dictionary: word_emission)
#The state transition probabilities P(S_i+1, S_i) are calculated (in dictionary: state_transition) with key as (S_i, S_i+1)
#The probability of the existence of a word in the train dataset is calculated (in dictionary: word_dict)
#The probability of the existence of a state in the train dataset is calculated (in dictionary: state_dict)
####

import random
import math
import numpy as np

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    
    def __init__(self):
        self.word_initial = {}
        self.word_emission = {}
        self.state_transition = {}
        self.word_dict = {}
        self.state_dict = {}
        
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        
        return 0

    # Do the training!
    #
    def train(self, data):
        
        for line in data:
            words = line[0]
            state = line[1]

            if len(words) != len(state):
                continue
            else:
                count = len(words)
                for i in range(0, count):
                    
                    #For the first word or state
                    if i == 0:
                        #add the word to the word_initial dictionary P(S_1)
                        if words[i] not in self.word_initial:
                            self.word_initial[words[i]] = 1
                        else:
                            self.word_initial[words[i]] += 1

                    #add the word to the word dictionary
                    if words[i] not in self.word_dict:
                        self.word_dict[words[i]] = 1
                    else:
                        self.word_dict[words[i]] += 1

                    #add the state to the state dictionary
                    if state[i] not in self.state_dict:
                        self.state_dict[state[i]] = 1
                    else:
                        self.state_dict[state[i]] += 1
                     
                    #add the state transition probabilities P(S_i+1, S_i) stored as (S_i, S_i+1) in the dictionary
                    if (i+1) != count:
                        if (state[i], state[i + 1]) not in self.state_transition:
                            self.state_transition[(state[i], state[i + 1])] = 1
                        else:
                            self.state_transition[(state[i], state[i + 1])] += 1
                    else:
                        if (state[i], "END") not in self.state_transition:
                            self.state_transition[(state[i], "END")] = 1
                        else:
                            self.state_transition[(state[i], "END")] += 1
                    
                    #add the emission probabilites P(W_i, S_i)
                    if (words[i], state[i]) not in self.word_emission:
                        self.word_emission[(words[i], state[i])] = 1
                    else:
                        self.word_emission[(words[i], state[i])] += 1
                            
                    
        #print 'Words: ', words
        #print 'State: ', state
        
        #print '\nWORD DICTIONARY: \n', self.word_dict
        #print '\nSTATE DICTIONARY: \n', self.state_dict
        #print '\nWORD INITIAL: \n', self.word_initial
        #print '\nSTATE TRANSITION \n', self.state_transition
        #print '\nWORD EMISSION \n', self.word_emission
        
        #converting to probability:
        self.word_dict = dict((key, float(value) / sum(self.word_dict.values())) for (key, value) in self.word_dict.items())
        self.state_dict = dict((key, float(value) / sum(self.state_dict.values())) for (key, value) in self.state_dict.items())
        self.word_initial = dict((key, float(value) / len(data)) for (key, value) in self.word_initial.items())
        self.state_transition = dict((key, float(value) / sum(self.state_transition.values())) for (key, value) in self.state_transition.items())
        self.word_emission = dict((key, float(value) / sum(self.word_emission.values())) for (key, value) in self.word_emission.items())
        
        '''
        for word in self.word_dict:
            if word not in self.word_initial:
                self.word_initial[word] = 0.0

            for state in self.state_dict:
                for state2 in self.state_dict:
                    if (state, state2) not in self.state_transition:
                        self.state_transition[(state, state2)] = 0.0
                                              
                if (word, state) not in self.word_emission:
                    self.word_emission[(word, state)] = 0.0
        '''
                               
        #print '\nWORD PROB: \n', self.word_dict
        #print '\nSTATE PROB: \n', self.state_dict
        #print '\nWORD INITIAL PROB: \n', self.word_initial
        #print '\nSTATE TRANSITION PROB \n', self.state_transition
        #print '\nWORD EMISSION PROB \n', self.word_emission
        
        pass

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        sequence = []
        for word in sentence:
            max_prob = 0
            max_prob_arg = ''
            for state, value in self.state_dict.items():
                #denominator can be removed for faster computation
                if (word, state) in self.word_emission:
                    prob = float(self.word_emission[(word, state)] * value) / float(self.word_dict[word])
                    if prob > max_prob:
                        max_prob = prob
                        max_prob_arg = state
            sequence.append(max_prob_arg)
            
        return sequence
        #return ["noun"] * len(sentence)

    def hmm_ve(self, sentence):
        sequence = []
        states = self.state_dict.keys()
        
        var_el = np.zeros((len(states), len(sentence)))

        for i in range(len(sentence)):
            word = sentence[i]
            for j in range(len(states)):
                state = states[j]

                if i == 0:
                    '''
                    if word in self.word_initial and (word, state) in self.word_emission:
                        var_el[j, 0] = self.word_initial[word] * self.word_emission[(word, state)]
                    else:
                        var_el[j, 0] = 0.0
                    '''
                    if (word, state) in self.word_emission:
                        if word in self.word_initial:
                            var_el[j, 0] = self.word_initial[word] * self.word_emission[(word, state)]
                        else:
                            var_el[j, 0] = self.word_emission[(word, state)]
                    else:
                        var_el[j, 0] = 0.0
                else:
                    temp_sum = 0.0
                    for k in range(len(states)):
                        state_prev = states[k]
                        if (state_prev, state) in self.state_transition:
                            temp_sum += (var_el[k, i - 1] * self.state_transition[(state_prev, state)])
                    if (word, state) in self.word_emission:
                        var_el[j, i] = temp_sum * self.word_emission[(word, state)]
                    else:
                        var_el[j, i] = 0.0

            max_prob_arg = np.argmax(var_el[:, i])
            sequence.append(states[max_prob_arg])
        
        return sequence
        #return ["noun"] * len(sentence)
        
    def hmm_viterbi(self, sentence):
        sequence = []
        states = self.state_dict.keys()
        
        var_el = np.zeros((len(states), len(sentence)))

        for i in range(len(sentence)):
            word = sentence[i]
            for j in range(len(states)):
                state = states[j]

                if i == 0:
                    '''
                    if word in self.word_initial and (word, state) in self.word_emission:
                        var_el[j, 0] = self.word_initial[word] * self.word_emission[(word, state)]
                    else:
                        var_el[j, 0] = 0.0
                    '''
                    if (word, state) in self.word_emission:
                        if word in self.word_initial:
                            var_el[j, 0] = self.word_initial[word] * self.word_emission[(word, state)]
                        else:
                            var_el[j, 0] = self.word_emission[(word, state)]
                    else:
                        var_el[j, 0] = 0.0
                else:
                    max_prob = 0.0
                    for k in range(len(states)):
                        state_prev = states[k]

                        if (state_prev, state) in self.state_transition:
                            prob = (var_el[k, i - 1] * self.state_transition[(state_prev, state)])
                        else:
                            prob = 0.0
                            
                        if prob > max_prob:
                            max_prob = prob
                                
                    if (word, state) in self.word_emission:
                        var_el[j, i] = max_prob * self.word_emission[(word, state)]
                    else:
                        var_el[j, i] = 0.0

            max_prob_arg = np.argmax(var_el[:, i])
            sequence.append(states[max_prob_arg])
        
        return sequence
        #return ["noun"] * len(sentence)

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print "Unknown algo!"

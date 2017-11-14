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
'''
In the train method, we have calculated the prior probabilities of all words and states in the training set.
The word emission probabilities P(W_i, S_i) are calculated (in dictionary: word_emission).
The state transition probabilities P(S_i+1, S_i) are calculated (in dictionary: state_transition) with key as (S_i, S_i+1).
The probability of the existence of a word in the train dataset is calculated (in dictionary: word_dict).
The probability of the existence of a state in the train dataset is calculated (in dictionary: state_dict).
The posterior probability P(W | S) = P(W, S) / P(S)
where P(W, S) = {P(S1) * P(W1 | S1) * P(S2 | S1) * P(W2 | S2) * P(S3 | S2) * P(W3 | S3) ...}
on taking the log,
log{P(W | S)} = log{P(W, S)} - log{P(S)}
log{P(W | S)} = {log P(S1) + log P(W1 | S1) + log P(S2 | S1) + log P(W2 | S2) + log P(S3 | S2) + ...} - log{P(S)}
If any combination of transition probabaility or emission probability is missing then we have assumed it to be equal to 0.00000000000000000000000000000001: a random value.
''' 
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
        self.word_emission = {}
        self.state_transition = {}
        self.word_dict = {}
        self.state_dict = {}
        
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        logB = 10
        
        N = len(sentence)
        sum = 0.0
        state_prev = None
        
        for i in range(0, N):
            word = sentence[i]
            state = label[i]
            
            if i == 0:
                if word in self.word_dict:
                    val = self.word_dict[word]
                else:
                    val = 0.00000000000000000000000000000001
                sum += math.log(val, logB)
            else:
                if (word, state_prev) in self.word_emission:
                    val = self.state_transition[(state_prev, state)]
                else:
                    val = 0.00000000000000000000000000000001
                sum += math.log(val, logB)
            
            if (word, state) in self.word_emission:
                val = self.word_emission[(word, state)]
            else:
                val = 0.00000000000000000000000000000001
            sum += math.log(val, logB)
            
            state_prev = label[i]
        
        return sum

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
                                                  
                    #add the emission probabilites P(W_i | S_i)
                    if (words[i], state[i]) not in self.word_emission:
                        self.word_emission[(words[i], state[i])] = 1
                    else:
                        self.word_emission[(words[i], state[i])] += 1
        
        #converting to probability:
        self.word_dict = dict((key, (float(value) / float(sum(self.word_dict.values())))) for (key, value) in self.word_dict.items())
        self.state_dict = dict((key, (float(value) / float(sum(self.state_dict.values())))) for (key, value) in self.state_dict.items())
        
        self.state_transition = dict((key, float(value) / float(sum(self.state_transition.values()))) for (key, value) in self.state_transition.items())
        self.word_emission = dict((key, float(value) / float(sum(self.word_emission.values()))) for (key, value) in self.word_emission.items())
        
        #To add the probabilities of events that were not encountered in the training dataset
        for word in self.word_dict:
            for state1 in self.state_dict:
                #state_transitions that were not in the train dataset
                for state2 in self.state_dict:
                    if (state1, state2) not in self.state_transition:
                        self.state_transition[(state1, state2)] = 0.00000000000000000000000000000001
                                              
                #emission probabilities that were not in the train dataset
                if (word, state1) not in self.word_emission:
                    self.word_emission[(word, state1)] = 0.00000000000000000000000000000001
        
        pass

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        sequence = []
        for word in sentence:
            max_prob = 0
            max_prob_arg = ''
            for state, value in self.state_dict.items():
                if (word, state) in self.word_emission:
                    #since the denominator is common for all the states, we can skip it: equal to the prior probability of the word
                    prob = value * self.word_emission[(word, state)]
                    if prob >= max_prob:
                        max_prob = prob
                        max_prob_arg = state
            sequence.append(max_prob_arg)
            
        return sequence

    def hmm_ve(self, sentence):
        sequence = []
        states = self.state_dict.keys()
        
        var_el = np.zeros((len(states), len(sentence)))

        for i in range(len(sentence)):
            word = sentence[i]
            
            for j in range(len(states)):
                state = states[j]
                
                if i == 0:
                    if (word, state) in self.word_emission:
                        if state in self.state_dict:
                            var_el[j, 0] = self.state_dict[state] * self.word_emission[(word, state)]
                        else:
                            #if the current state was never encountered in the training set
                            var_el[j, 0] = 0.00000000000000000000000000000001
                    else:
                        #if the current (word, state) pair was never encountered in the training set
                        var_el[j, 0] = 0.00000000000000000000000000000001
                else:
                    temp_sum = 0.0
                    #k is for previous state
                    for k in range(len(states)):
                        state_prev = states[k]
                        if (state_prev, state) in self.state_transition:
                            temp_sum += (var_el[k, i - 1] * self.state_transition[(state_prev, state)])
                            
                    if (word, state) in self.word_emission:
                        var_el[j, i] = temp_sum * self.word_emission[(word, state)]
                    else:
                        #if the current (word, state) pair was never encountered in the training set
                        var_el[j, i] = temp_sum * 0.00000000000000000000000000000001

            max_prob_arg = np.argmax(var_el[:, i])
            sequence.append(states[max_prob_arg])
        
        return sequence
        
    def hmm_viterbi(self, sentence):
        sequence = []
        states = self.state_dict.keys()
        
        var_el = np.zeros((len(states), len(sentence)))

        for i in range(len(sentence)):
            word = sentence[i]
            
            if i == 0:
                for j in range(len(states)):
                    state = states[j]
      
                    if (word, state) in self.word_emission:
                        if state in self.state_dict:
                            var_el[j, i] = self.state_dict[state] * self.word_emission[(word, state)]
                        else:
                            #if the current state was never encountered in the training set
                            var_el[j, i] = 0.00000000000000000000000000000001
                    else:
                        #if the current (word, state) pair was never encountered in the training set
                        var_el[j, 0] = 0.00000000000000000000000000000001
            else:
                for j in range(len(states)):
                    state = states[j]

                    max_prob = 0.0
                    for k in range(len(states)):
                        state_prev = states[k]
    
                        if (state_prev, state) in self.state_transition:
                            prob = (var_el[k, i - 1] * self.state_transition[(state_prev, state)])
                            
                        if prob > max_prob:
                            max_prob = prob
                                
                    if (word, state) in self.word_emission:
                        var_el[j, i] = max_prob * self.word_emission[(word, state)]
                    else:
                        #if the current (word, state) pair was never encountered in the training set
                        var_el[j, i] = max_prob * 0.00000000000000000000000000000001

            max_prob_arg = np.argmax(var_el[:, i])
            sequence.append(states[max_prob_arg])
        
        return sequence

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

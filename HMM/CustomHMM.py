import torch
import numpy as np
import pdb
import datetime
import torch.nn.functional as f
import matplotlib.pyplot as plt
import os
import traceback
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle

torch.autograd.set_detect_anomaly(True)

histones = ["H2AK5ac","H2AS129ph","H3K14ac","H3K18ac","H3K23ac","H3K27ac","H3K36me","H3K36me2","H3K36me3","H3K4ac","H3K4me","H3K4me2","H3K4me3","H3K56ac","H3K79me","H3K79me3","H3K9ac","H3S10ph","H4K12ac","H4K16ac","H4K20me","H4K5ac","H4K8ac","H4R3me","H4R3me2s","Htz1"]

# a method which returns the relevant matricies learned from the standard HMM
# I generate/format these in the Standard HMM Code
# if cross_fold is 0, then its not based on a crossfold but parameters from a model trained with all of the data
def get_premade_matricies(s, cross_fold):
    s = str(s)
    cross_fold = str(cross_fold)
    
    INITAL_T = pickle.load("hmmmatrix_t_" + s + "states_" + cross_fold + "fold.pkl")
    INITAL_E = pickle.load("hmmmatrix_e_" + s + "states_" + cross_fold + "fold.pkl")
    INITAL_T0 = pickle.load("hmmmatrix_t0_" + s + "states_" + cross_fold + "fold.pkl")

    return INITIAL_T, INITIAL_E, INITIAL_T0

# begin HMM class definition
class HiddenMarkovModel(object):
    """
    Hidden Markov self Class

    Parameters:
    -----------
    
    - S: Number of states.
    - T: numpy.array Transition matrix of size S by S
         stores probability from state i to state j.
    - E: numpy.array Emission matrix of size S by N (number of observations)
         stores the probability of observing  O_j  from state  S_i. 
    - T0: numpy.array Initial state probabilities of size S.
    """

    # initialize the model class
    def __init__(self, T, E, T0, representedHistones, epsilon = 0.001, maxStep = 10):
        # Max number of iteration
        self.maxStep = maxStep
        # convergence criteria
        self.epsilon = epsilon 
        # Number of possible states
        self.S = T.shape[0]
        # Number of possible observations
        self.O = E.shape[0]
        self.prob_state_1 = []
        # Emission probability
        self.E = E.clone()
        # Transition matrix
        self.T = T.clone()
        # Initial state vector
        self.T0 = T0.clone()

        # for generating the log likelihood plots
        self.scale_plot = list()

        # for generating the log likelihood plots
        self.converged_at = 0

        # making the error a variable global to the model
        # initializing it low, so that the first step's error is always better
        self.error = torch.tensor([-1000000000000], requires_grad=True, dtype=torch.float64)
        self.old_test_seq_likelihood = -1000000000000

        self.validation_error = torch.tensor([-1000000000000], requires_grad=True, dtype=torch.float64)

        gw_temp = torch.zeros(5, T.shape[0], T.shape[0], dtype=torch.float64)
        gw_temp[4] = torch.log(self.T)

        # initializing the genomic weights to zero. The bias term is the stabilized self.T
        self.genomic_weights = gw_temp.clone().requires_grad_(True)

        self.original_genomic_weights = self.genomic_weights.clone()

        # initializing the learning rate
        self.original_learning_rate = 0.001 #0.001 205
        self.learning_rate =  self.original_learning_rate #0.001 #0.0002

        # initialized to a proper value at the end of each descent step
        self.old_genomic_weights = None
        #self.old_test_seq_likelihood = None

        self.representedHistones = representedHistones

    # T = the original transition matrix
    # xVal = count of base type in assosciated genomic region
    # returns an adjusted transition matrix, based on the genomic values
    # this method is new (ie doesn't appear in the original library)
    def adjust_T(self, genomic_values):

        aVal = self.avgA
        tVal = self.avgT
        cVal = self.avgC
        gVal = self.avgG

        # create a vector of the base frequencies for the current region
        self.genomic_values = torch.tensor([(aVal - self.avgA), (tVal - self.avgT), (cVal - self.avgC), (gVal - self.avgG)], dtype=torch.float64)

        # adjusting T
        self.returnT = torch.exp((self.genomic_weights[0] * self.genomic_values[0]) + (self.genomic_weights[1] * 
            self.genomic_values[1]) + (self.genomic_weights[2] * self.genomic_values[2]) + (self.genomic_weights[3] * 
            self.genomic_values[3]) + self.genomic_weights[4])
        
        # normalizing
        norm = self.returnT.norm(p=1, dim=1, keepdim=True)
        self.returnT = self.returnT.div(norm.expand_as(self.returnT))

        # return
        return self.returnT
    
    # this is a method from the library, the only change I've made is changing 
    # the tensor data type to prevent underflow/overflow
    def initialize_viterbi_variables(self, shape): 
        pathStates = torch.zeros(shape, dtype=torch.float64)
        pathScores = torch.zeros_like(pathStates, dtype=torch.float64)
        states_seq = torch.zeros([shape[0]], dtype=torch.int64)
        return pathStates, pathScores, states_seq
    
    # viterbi_inference is used to get the most likely path sequence for the training data, 
    # so belief_propogration must take the adjusted T into account
    def belief_propagation(self, scores, genome_data):
        return scores.view(-1,1) + torch.log(self.adjust_T(genome_data))
    
    # this method is mostly unchanged from the original library, 
    # except for adjusting the transition matrix at each step
    def viterbi_inference(self, x, observation_genome_data): # x: observing sequence        
        self.N = len(x)
        shape = [self.N, self.S]

        # Init_viterbi_variables
        pathStates, pathScores, states_seq = self.initialize_viterbi_variables(shape) 

        # log probability of emission sequence
        obs_prob_full = torch.log(self.generate_obs_prob_seq(x))

        # initialize with state starting log-priors
        pathScores[0] = torch.log(self.T0) + obs_prob_full[0]

        genome_data_index = 0

        for step, obs_prob in enumerate(obs_prob_full[1:]):
            # propagate state belief
            belief = self.belief_propagation(pathScores[step, :], observation_genome_data[genome_data_index])
            # the inferred state by maximizing global function
            pathStates[step + 1] = torch.argmax(belief, 0)
            # and update state and score matrices 
            pathScores[step + 1] = torch.max(belief, 0)[0] + obs_prob

            # increment index of genome data list
            genome_data_index = genome_data_index + 1

        # infer most likely last state
        states_seq[self.N - 1] = torch.argmax(pathScores[self.N-1, :], 0)
        for step in range(self.N - 1, 0, -1):
            # for every timestep retrieve inferred state
            state = states_seq[step]
            state_prob = pathStates[step][state]
            states_seq[step -1] = state_prob

        return states_seq, torch.exp(pathScores) # turn scores back to probabilities
    
    # This method is unchanged from the original library, 
    # except for changing data types
    def initialize_forw_back_variables(self, shape):
        self.forward = torch.zeros(shape, dtype=torch.float64)
        self.backward = torch.zeros_like(self.forward, dtype=torch.float64)
        self.posterior = torch.zeros_like(self.forward, dtype=torch.float64)
        
    # this method is mostly unchanged from the original library, 
    # except for adjusting the transition matrix at each step,
    # and for saving the scaling factor at each step for later use
    def _forward(model, obs_prob_seq, save_scale_plot):
        model.scale = torch.zeros([len(obs_prob_seq)], dtype=torch.float64) #scale factors
        
        # initialize with state starting priors
        init_prob = model.T0 * obs_prob_seq[0]
        # scaling factor at t=0
        model.scale[0] = 1.0 / init_prob.sum()

        # scaled belief at t=0
        #model.forward[0] = model.scale[0] * init_prob
        model.forward[0] = torch.nn.functional.normalize(init_prob, p=1, dim=0)

         # propagate belief
        for step, obs_prob in enumerate(obs_prob_seq[1:]):
            # get adjusted transition matrix for nucleosome
            adjusted_T = model.adjusted_Ts[step + 1] #model.adjust_T(genome_data[step])

            # previous state probability
            prev_prob = model.forward[step].unsqueeze(0).clone()

            # transition prior
            prior_prob = torch.matmul(prev_prob, adjusted_T)

            # forward belief propagation
            forward_score = prior_prob * obs_prob
            forward_prob = torch.squeeze(forward_score).clone()

            # scaling factor
            model.scale[step + 1] = 1 / forward_prob.sum()

            # Update forward matrix
            model.forward[step + 1] = model.scale[step + 1].clone() * forward_prob
    
    # this method is mostly unchanged from the original library, 
    # except for adjusting the transition matrix at each step,
    # and normalizing to prevent underflow/overflow
    def _backward(self, obs_prob_seq_rev):

        # initialize with state ending priors
        self.backward[0] = self.scale[len(obs_prob_seq_rev) - 1] * torch.ones([self.S], dtype=torch.float64)

        # propagate belief
        for step, obs_prob in enumerate(obs_prob_seq_rev[:-1]):

            # generate adjusted transition matrix for nucleosome
            adjusted_T = self.adjusted_Ts[(len(obs_prob_seq_rev) - step) - 1]

            # next state probability
            next_prob = self.backward[step, :].unsqueeze(1).clone()

            # observation emission probabilities
            obs_prob_d = torch.diag(obs_prob)

            # transition prior
            prior_prob = torch.matmul(adjusted_T, obs_prob_d)

            # backward belief propagation
            backward_prob = torch.matmul(prior_prob, next_prob).squeeze()

            # Update backward matrix
            self.backward[step + 1] = self.scale[len(obs_prob_seq_rev) - 1 - step] * backward_prob

        self.backward = torch.flip(self.backward, [0])

    # this method is mostly unchanged from the original library, 
    # except for piping through the relevant genome_data so that the
    # transition matrix can be adjusted at each step
    def forward_backward(self, obs_prob_seq, genome_data, save_scale_plot):
        """
        runs forward backward algorithm on observation sequence

        Arguments
        ---------
        - obs_prob_seq : matrix of size N by S, where N is number of timesteps and
            S is the number of states

        Returns
        -------
        - forward : matrix of size N by S representing
            the forward probability of each state at each time step
        - backward : matrix of size N by S representing
            the backward probability of each state at each time step
        - posterior : matrix of size N by S representing
            the posterior probability of each state at each time step
        """        
        self.adjusted_Ts = dict()

        for i in range(len(obs_prob_seq)):
            self.adjusted_Ts[i] = self.adjust_T(genome_data[i])

        self._forward(obs_prob_seq, save_scale_plot)
        
        obs_prob_seq_rev = torch.flip(obs_prob_seq, [0])
       
        self._backward(obs_prob_seq_rev)

    # this method is new
    # checks convergence of genomic weight values
    def check_genomic_weight_convergence(self, new_genomic_weights):
        # check to see if change in genomic weight values is less than the epsilon
        delta_GW = torch.max(torch.abs(self.genomic_weights - new_genomic_weights)).item() < self.epsilon

        # print the delta, so I can monitor how close to convergence its getting
        print("delta_GW (must be < " + str(self.epsilon) + " for convergence) = " + str(torch.max(torch.abs(self.genomic_weights - new_genomic_weights)).item()))

        if(self.learning_rate < 0.000000000000000000001):
          return True

        # returns true/false
        return delta_GW

    # generates a list of the possible transition matrix values for each emission seen in the training sequence
    # this method is new
    def generate_obs_prob_seq(self, obs_seq, index_to_ignore):
        if index_to_ignore < 1:
            index_to_ignore = 5000000000

        # initialize the returned matrix
        obs_prob_seq = torch.zeros(len(obs_seq), self.T.shape[0], requires_grad=False, dtype=torch.float64)

        # for each observation
        for a in range(len(obs_seq)):
            obs = obs_seq[a]

            # generate the probability that it came from a particular state
            for i in range(self.T.shape[0]):
                totprob = 1
                state = i

                # for each modification in the emission vector
                for j in range(len(obs) - 1):
                    if j != index_to_ignore:
                        mod = int(obs[j])
                        place = j

                        # the likelihood the mod is 1 is given by self.E[place][state]
                        if mod == 1:
                            totprob = totprob * self.E[place][state]
                        # the likelihood the mod is 0 is given by 1 - self.E[place][state]
                        else:
                            totprob = totprob * (1 - self.E[place][state])

                obs_prob_seq[a][state] = totprob

        for x in range(len(obs_prob_seq)):
            row = obs_prob_seq[x]

            rowSum = row.sum()

            if rowSum == 0:
                for y in range(len(row)):
                    obs_prob_seq[x][y] =  1 / len(row)q 514

        return obs_prob_seq

    # this is the method that performs each learning step
    # it takes the sequence of observation data, and the sequence of genome data
    # it returns true or false indicating if convergence has been reached
    # this method is new, but is formatted based on the EM method from the original library
    def descent(self, obs_seq, genome_data, test_seq, test_seq_genome_data):

        # probability of emission sequence
        obs_prob_seq = self.generate_obs_prob_seq(obs_seq, -1)

        # run forward backward
        self.forward_backward(obs_prob_seq, genome_data, True)

        # save the previous error for comparison
        prev_error = self.error

        # calculate the error
        self.error = self.get_likelihood_of_seq(obs_seq, genome_data)

        self.scale_plot.append(self.error)

        self.prev_validation_error = self.validation_error
        self.validation_error = self.get_likelihood_of_seq(test_seq, test_seq_genome_data)

        # print the error
        print("training seq log likelihood = " + str(self.error))

        # otherwise pytorch doesn't save old computation graph, which makes it unable to compute grad
        self.genomic_weights.retain_grad()

        prevTestLikelihood = self.test_seq_likelihood

        self.test_seq_likelihood = self.get_likelihood_of_seq(test_seq, test_seq_genome_data)

        cont=False

        final_likelihood = self.get_likelihood_of_seq(self.final_test, self.final_test_data)

        print("final test seq likelihood="+str(final_likelihood))

        # if the error has gotten worse, instead of better. Error here is -1 * log likelihood
        if self.error != self.error or self.error < prev_error or prevTestLikelihood > self.test_seq_likelihood:
        #if self.validation_error < self.prev_validation_error:
            print("trainging seq LL of " + str(self.error) + " is worse than " + str(prev_error)+" at the previous descent step, backing off and trying again")
           
            # back off on the learning rate as usual
            self.learning_rate = self.learning_rate * 0.5

            # revert to previous genomic weights (if possible), and try again!
            self.genomic_weights = self.old_genomic_weights
            
            self.test_seq_likelihood = prevTestLikelihood

            self.error = prev_error

            # this is needed because if you checked for convergence normally, it would 
            # always return true since nothing is changing
            converged = False

        else:
          # backward step based on error
          self.error.backward(retain_graph=True)
          cont = True

        # checking to see if gradient is valid
        if(self.genomic_weights.grad is not None) and cont:
            # save the previous genomic weights values
            # cloned to detach it from the computation graph
            self.old_genomic_weights = self.genomic_weights.clone()
            self.old_test_seq_likelihood = self.get_likelihood_of_seq(test_seq, test_seq_genome_data)

            # adjust genomic weights based on gradient/learning rate
            self.genomic_weights = self.genomic_weights + (self.learning_rate * self.genomic_weights.grad)
            # back off on learning rate
            self.learning_rate = self.learning_rate * 0.5

            # if self.old_genomic_weights is not set, then its the first step
            if(self.old_genomic_weights is not None):
                # check for convergence
                converged = self.check_genomic_weight_convergence(self.old_genomic_weights)
            else:
                converged = False

        elif(cont):
            # something went wrong with calculating the gradient
            # sad
            print(":'(")

        print("new learning rate = " + str(self.learning_rate))
        
        # return whether or not the genomic_weights have converged
        return converged

    # gets likelihood of a test sequence
    # it takes an observation sequence and sequence of genomic data as parameters
    # it returns the log likelihood of the sequence passed in
    # this method is new
    def get_likelihood_of_seq(self, obs_seq, genome_data):

        torch.set_printoptions(profile="full")

        # probability of emission sequence
        obs_prob_seq = self.generate_obs_prob_seq(obs_seq, -1)

        # shape of Variables
        shape = [len(obs_seq), self.S]

        # initialize variables
        self.initialize_forw_back_variables(shape)

        # run forward backward once
        self.forward_backward(obs_prob_seq, genome_data, False)

        forward_prob = self.forward[-1]
        backward_prob = self.backward[-1]

        product = torch.dot(forward_prob, backward_prob)

        otherLikelihood = -1 * torch.log(torch.sum(product)) * torch.sum(torch.log(self.scale))

        # reset variables
        shape = [self.N, self.S]
        self.initialize_forw_back_variables(shape)

        return otherLikelihood

    # get the index of a matrix row which has the highest value
    # this method is new
    def get_max_index(self, the_list):
        index = 0
        maxScore = 0

        # for each item in the row
        for i in range(len(the_list)):

            # if the value is greater than the previous max, save it as the new max
            if the_list[i] > maxScore:
                index = i
                maxScore = the_list[i]

        # return the index corresponding to the max value
        return index

    # utility method to make a biased random guess based on probabilities
    # this method is new
    def get_guess(self, the_list):
        return np.random.choice(np.arange(0, len(the_list)), p=the_list.detach().numpy())

    # utility method to make a biased random guess of whether a vector entry should be 1/0
    # this method is new
    def guess_emission(self, probability):
        the_list = [(1-probability), probability]

        return np.random.choice(np.arange(0, 2), p=the_list)

    # utility for saving plots
    # this method is new
    def save_figure(self, chromosome, test_num, random, cross_fold):
        # plot log likelihoods of training data

        # if it reached max step instead of converging, you don't want to add one
        x_axis_count = self.converged_at + 1
        if x_axis_count == self.maxStep + 1:
            x_axis_count = self.maxStep

        # x axis if the # of steps
        x_axis = range(len(self.scale_plot))
        # y axis is the log likelihood of the training sequence at each step
        y_axis = self.scale_plot

        x_axis = np.asarray(x_axis)
        y_axis = np.asarray(y_axis)

        # plot
        plt.plot(x_axis, y_axis)
        plt.xlabel('Step Number')
        plt.ylabel('Log Likelihood')

        chrom_name = chromosome[0: -1]

        # for docker builds using environmental variables to set the output location
        #plt.savefig(os.environ['OUTPUT_PATH']+"/figures/chrom_" + chrom_name + "_states_" + str(self.S) + "_testnum_" + str(test_num) + ".png")
        
        # for running locally
        plt.savefig("./testing_likelihood_at_each_step/reallyjustgc_custom_chrom_" + chrom_name + "_states_" + str(self.S) + "_testnum_" + str(test_num) + "_crossfold_" + str(cross_fold) +"_allhistonestested.png")

        # start new plot
        plt.clf()

        # plot log likelihoods of test data

        # x axis is the # of steps
        x_axis = range(len(self.test_seq_likelihoods))
        # y axis is the test seq likelihoods
        y_axis = self.test_seq_likelihoods

        x_axis = np.asarray(x_axis)
        y_axis = np.asarray(y_axis)

        # plot
        plt.plot(x_axis, y_axis)
        plt.xlabel('Step Number')
        plt.ylabel('Test Seq Log Likelihood')


        # for docker builds using environmental variables to set the output location
        #plt.savefig(os.environ['OUTPUT_PATH']+"/figures/chrom_" + chrom_name + "_states_" + str(self.S) + "_testnum_" + str(test_num) + ".png")
        
        # for running locally
        plt.savefig("./testing_likelihood_at_each_step_likelihoods/reallyjustgc_custom_chrom_" + chrom_name + "_states_" + str(self.S) + "_testnum_" + str(test_num) + "_crossfold_" + str(cross_fold) +"_allhistonestested.png")

        # start a new plot
        plt.clf()

        # plot accuracies in predicting test sequence

        self.multiplot(self.accuracies, "accuracy", chrom_name, test_num, cross_fold)
        self.multiplot(self.recalls, "recall", chrom_name, test_num, cross_fold)
        self.multiplot(self.precisions, "precision", chrom_name, test_num, cross_fold)
        
        # so I can use them later if needed
        print("accuracies")
        print(self.accuracies)
        print("recalls")
        print(self.recalls)
        print("precisions")
        print(self.precisions)

    def multiplot(self, dictionary, name, chrom_name, test_num, cross_fold):
        # plot dictionary in predicting test sequence

        # x axis is the # of steps
        x_axis = range(len(dictionary[list(dictionary.keys())[0]]))
        x_axis = np.asarray(x_axis)
        for key in dictionary.keys():
            # y axis is the test seq likelihoods
            y_axis = dictionary[key]
            y_axis = np.asarray(y_axis)

            # plot
            plt.plot(x_axis, y_axis, label=histones[key])
            plt.xlabel('Step Number')
            plt.ylabel('Validation Seq Prediction ' + name)


        # for docker builds using environmental variables to set the output location
        #plt.savefig(os.environ['OUTPUT_PATH']+"/figures/chrom_" + chrom_name + "_states_" + str(self.S) + "_testnum_" + str(test_num) + ".png")
        
        plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)

        # for running locally
        plt.savefig("./testing_likelihood_at_each_step_"+name+"/reallyjustgc_custom_chrom_" + chrom_name + "_states_" + str(self.S) + "_testnum_" + str(test_num) + "_crossfold_" + str(cross_fold) +"_allhistonestested.png", bbox_inches="tight")

        # start a new plot
        plt.clf()

    def posterior_decoding(self, seq, genome_data, index_to_ignore):
        # probability of emission sequence
        obs_prob_seq = self.generate_obs_prob_seq(seq, index_to_ignore)

        # shape of Variables
        shape = [len(seq), self.S]

        # initialize variables
        self.initialize_forw_back_variables(shape)

        # run forward backward once
        self.forward_backward(obs_prob_seq, genome_data, False)

        # compute state sequence via posterior decoding
        state_seq = []

        # for each timestep
        for i in range(len(seq)):
            scores = self.forward[i] * self.backward[i]
            state = self.get_max_index(scores)
            state_seq.append(state)

        # reset variables
        shape = [self.N, self.S]
        self.initialize_forw_back_variables(shape)

        return state_seq

    def getStatsForLogReg(self, observations, training_sequence_genome_data, test_seq, test_sequence_genome_data):
        accuracies = {}
        precisions = {}
        recalls = {}

        for test_index in range(len(observations[0])):
            if test_index in self.representedHistones.keys():
                guessed_sequence = self.guess_sequence_logreg(observations, training_sequence_genome_data, test_seq, test_sequence_genome_data, 3, test_index)
                accuracy, precision, recall = generate_stats(guessed_sequence, test_seq, test_index)
                accuracies[test_index] = accuracy
                precisions[test_index] = precision
                recalls[test_index] = recall

        return accuracies, precisions, recalls



    def guess_sequence_logreg(self, training_sequence, training_sequence_genome_data, test_sequence, test_sequence_genome_data, pred_type, histone_index_to_test):
        # generating states of TEST sequence (ASK SHAY IF THAT MAKES SENSE!!)

        new_test_seq = test_sequence.copy()

        for i in range(len(new_test_seq)):
            new_test_seq[i] = new_test_seq[i][:histone_index_to_test] + new_test_seq[i][(histone_index_to_test + 1):]

        # now we want to use logistic regression to predict one histone modification for the test sequence
        # pred_type=1 -> only ATCG
        # pred_type=2 -> only state
        # pred_type=3 -> both

        use_state_seq = False
        use_ATCG = False
        if pred_type == 2 or pred_type == 3:
            use_state_seq = True

        if pred_type == 1 or pred_type == 3:
            use_ATCG = True

        save_genomic_weights = self.genomic_weights.clone()
        self.genomic_weights = self.original_genomic_weights

        test_state_seq = self.posterior_decoding(new_test_seq, test_sequence_genome_data, histone_index_to_test)
        train_state_seq = self.posterior_decoding(training_sequence, training_sequence_genome_data, -1)
        

        final_seq = []

        logregdata = {}
        if(use_state_seq):
            logregdata['state'] = []
            logregdata['emission_prob'] = []
        if(use_ATCG):
            logregdata['A'] = []
            logregdata['T'] = []
            logregdata['C'] = []
            logregdata['G'] = []

        emptyCount = 0

        # generate dictionary for logreg
        for x in range(len(training_sequence)):
            obs = training_sequence[x]

            empty = True
            for i in obs:
                if i == '1':
                    empty = False

            if empty:
                emptyCount = emptyCount + 1

            else:
                emptyCount = 0

            if not empty: # or emptyCount < 5:
                for foobar in range(1):
                    for i in range(len(obs)):
                        #if i == histone_index_to_test:
                        if str(i) not in logregdata:
                            logregdata[str(i)] = []

                        logregdata[str(i)].append(int(obs[i]))
                    if(use_state_seq):
                        logregdata['state'].append(train_state_seq[x])
                        logregdata['emission_prob'].append(self.E[histone_index_to_test][train_state_seq[x]])
                    if(use_ATCG):
                        logregdata['A'].append(training_sequence_genome_data[x][0])
                        logregdata['T'].append(training_sequence_genome_data[x][1])
                        logregdata['C'].append(training_sequence_genome_data[x][2])
                        logregdata['G'].append(training_sequence_genome_data[x][3])
        df = pd.DataFrame(logregdata,columns= logregdata.keys())


        columns = list(logregdata.keys())
        columns.remove(str(histone_index_to_test))

        X = df[columns]
        y = df[str(histone_index_to_test)]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0,random_state=0)
        logistic_regression = LogisticRegression()
        logistic_regression.fit(X_train,y_train)


        for index in range(len(test_state_seq)):

            # i is predicted state from posterior decoding
            i = test_state_seq[index]

            vector = new_test_seq[index]
            full_vector = test_sequence[index]
            char = "0"
            
            if self.E[histone_index_to_test][i] >= 1:
                char = "1"

            prediction_data = {}
            if(use_state_seq):
                prediction_data['state'] = [i]
                prediction_data['emission_prob'] = [self.E[histone_index_to_test][i]]
            if(use_ATCG):
                prediction_data['A'] = [training_sequence_genome_data[i][0]]
                prediction_data['T'] = [training_sequence_genome_data[i][1]]
                prediction_data['C'] = [training_sequence_genome_data[i][2]]
                prediction_data['G'] = [training_sequence_genome_data[i][3]]
            for h in range(len(full_vector)):
                if h != histone_index_to_test:
                    prediction_data[str(h)] = [int(full_vector[h])]

            X_test = pd.DataFrame(prediction_data,columns= prediction_data.keys())
            y_pred=logistic_regression.predict(X_test)

            char = str(y_pred[0])

            vector = vector[:histone_index_to_test] + char + vector[histone_index_to_test:]

            final_seq.append(vector)

        self.genomic_weights = save_genomic_weights

        return final_seq
    
    # method for setting up training
    # this method is new, but formatted based on the Baum_Welch_EM 
    # method from the library
    def learn_genomic_weights(self, obs_seq, genome_data, avgA, avgT, avgC, avgG, test_seq, test_seq_genome_data, test_index):
        self.error = torch.tensor([-1000000000000], requires_grad=True, dtype=torch.float64)
        self.old_test_seq_likelihood = -1000000000000
        self.learning_rate = self.original_learning_rate

        # length of observed sequence
        self.N = len(obs_seq)
        
        # initialize scale tensor
        self.scale = torch.zeros([self.N], dtype=torch.float64, requires_grad=False) #scale factors

        # set average base counts for whole training sequence
        self.avgA = avgA
        self.avgT = avgT
        self.avgC = avgC
        self.avgG = avgG

        # shape of Variables
        shape = [self.N, self.S]

        self.accuracies = {}
        self.precisions = {}
        self.recalls = {}

        # initialize variables
        self.initialize_forw_back_variables(shape)

        # some setup
        converged = False

        # initialize list of test sequence likelihoods
        self.test_seq_likelihoods = []

        # initialize variable which will store the learned matricies at each step of learning
        self.previous_transitions = []
        self.previous_emissions = []
        self.previous_t0s = []
        self.previous_genomic_weights = []

        initialLikelihood = self.get_likelihood_of_seq(obs_seq, genome_data)
        held_back_likelihood = self.get_likelihood_of_seq(test_seq, test_seq_genome_data)
        self.test_seq_likelihood = held_back_likelihood

        # clear scale plot, because we don't want this extra likelihood calculation included
        self.scale_plot = []

        self.scale_plot.append(initialLikelihood)
        self.test_seq_likelihoods.append(held_back_likelihood)
        self.previous_genomic_weights.append(self.genomic_weights.clone())

        print("training seq log likelihood before first descent step = " + str(initialLikelihood))
        print("test seq log likelihood before first descent step = " + str(held_back_likelihood))

        # for each step until convergence or the max step number
        for i in range(self.maxStep):

            # print the step number and time it took to run, also flush the 
            # output so there's no annoying buffering
            print("step "+ str(i) + " at " + str(datetime.datetime.now()), flush=True)

            # run gradient descent
            converged = self.descent(obs_seq, genome_data, test_seq, test_seq_genome_data)

            # save updated variables for this step
            self.previous_genomic_weights.append(self.genomic_weights.clone())

            # generate the likelihood of the test sequence based on this step's matricies
            held_back_likelihood = self.get_likelihood_of_seq(test_seq, test_seq_genome_data)

            # print test sequence likelihood
            print("testing seq log likelihood = " + str(held_back_likelihood))

            # add the test sequence's likelihood to the list of test sequence likelihoods, so we can plot later
            self.test_seq_likelihoods.append(held_back_likelihood)

            if(len(self.test_seq_likelihoods) > 1 and self.test_seq_likelihoods[-1] < self.test_seq_likelihoods[-2]):
                print("CURRENT TESTING SEQUENCE LL IS WORSE THAN @ PREVIOUS DESCENT STEP")

            # if the delta is less than the epsilon
            if converged and i != 0:

                # print step when I converged
                print('converged at step {}'.format(i))
                self.converged_at = i

                # break out of the loop, as we don't need to continue learning
                break 
            else:
                self.converged_at = self.maxStep

        # print the seqence of log likelihoods (in case I want to plot again later)
        print("scale plot")
        print(self.scale_plot)

        # print the list of test sequence log likelihoods (in case I want to plot again later)
        print("test seq likelihoods plot")
        print(self.test_seq_likelihoods)

        # get the matrices assosciated with the highest test sequence likelihood
        maxIndex = -1
        maxLikelihood = -1 * np.inf
        for x in range(len(self.test_seq_likelihoods)):
            if (self.test_seq_likelihoods[x] > maxLikelihood):
                maxLikelihood = self.test_seq_likelihoods[x]
                maxIndex = x

        # print out the best test sequence likelihood
        print("Max test seq likelihood= " + str(maxLikelihood))
        print("Max test seq likelihood index= " + str(maxIndex))

        self.genomic_weights = self.previous_genomic_weights[maxIndex]

        # return the last matricies
        return self.T0, self.T, self.E, self.genomic_weights, converged

# method for generating precision/recall/accuracy/f1 score
# this method is new
def generate_stats(guessed_sequence, held_observations, index_that_matters):

    # initialize return string, which will hold the formatted stats
    retString = ""

    # initialize some useful variables for computing the stats
    trueNegatives = 0
    truePositives = 0
    falseNegatives = 0
    falsePositives = 0
    total = 0

    # for each vector in the predicted emission sequence
    for i in range(len(guessed_sequence)):
        # give the guessed/actual sequences some more useful names
        guessed = guessed_sequence[i]
        actual = held_observations[i]

        # for each histone modification
        for h in range(len(guessed)):

            if h == index_that_matters:

                # determine if the predicted vector value is correct/incorrect
                if guessed[h] == "0" and actual[h] == "0":
                    trueNegatives = trueNegatives + 1
                elif guessed[h] == "1" and actual[h] == "1":
                    truePositives = truePositives + 1
                elif guessed[h] == "0" and actual[h] == "1":
                    falseNegatives = falseNegatives + 1
                elif guessed[h] == "1" and actual[h] == "0":
                    falsePositives = falsePositives + 1

    # compute precision (if possible)
    if (truePositives + falsePositives) != 0:
        precision = truePositives / (truePositives + falsePositives)
    else:
        #print("truePositives + falseNegatives = 0, precision cannot be calculated")
        precision = np.nan

    # compute recall (if possible)
    if (truePositives + falseNegatives) != 0:
        recall = truePositives / (truePositives + falseNegatives)
    else:
        #print("truePositives + falseNegatives = 0, recall cannot be calculated")
        recall = np.nan

    # compute f1 (if possible)
    if (precision + recall) != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        #print("precision + recall = 0, f1 cannot be calculated")
        f1 = np.nan

    # compute accuracy (if possible)
    accuracy = (truePositives + trueNegatives) / (trueNegatives + truePositives + falsePositives + falseNegatives)

    # format the return string
    retString = retString + ", " + str(trueNegatives) + ", " + str(truePositives) + ", " + str(falseNegatives) + ", " + str(falsePositives) + ", " + str(precision) + ", " + str(recall) + ", " + str(f1) + ", " + str(accuracy) + ", " + str((trueNegatives + truePositives + falsePositives + falseNegatives))

    # return
    return accuracy, precision, recall

def get_chunks(lst, n, folds):
    """Yield successive n-sized chunks from lst."""
    chunks = [lst[i * n:(i + 1) * n] for i in range((len(lst) + n - 1) // n )]  

    while(len(chunks) > folds):
        del chunks[-1]

    return chunks

# method to make running a bunch of experiments easier
# takes a chromosome, test ID number, state count, and a flag indicating 
# whether the matricies should come from the standard HMM, or be randomly 
# generated
# this method is new
def run_experiment(testChrom, test_num, state_count, random, cross_fold):
    test_index = 12

    # initialize the string which will eventually be a formatted data row in a CSV file
    data_string = testChrom[0:-1] + ", " + str(state_count) + ", " + str(test_num) + ", " + str(cross_fold)  

    # initialize list which will be the list of observations in the training set
    observations = []
    # initialize observation count
    count = 0
    # initialzie list which will represent the genome data for the observation sequence
    observation_genome_data = []

    # flag for determining if we've reached the correct point in the input file for the chromosome
    start = True 

    # initialize values which will represent the average A/T/C/G counts
    avgA = 0
    avgT = 0
    avgC = 0
    avgG = 0

    # open file with nucleosome vector sequences
    with open("yeast_vector_sequence_for_HMM.txt") as f:
        # for each like in the file
        for line in f:
            # if we've reached the beginning of the chromosome we're interested in
            if testChrom in line:
                #set flag indicating to start reading in the file
                start = True
                
            # if its the right kind of line, and the start flag has been set to true, start processing
            if "K36me" not in line and start and "chr" not in line and ("0" in line or "1" in line) and "{" not in line:
                # increment the observation count
                count = count + 1

                # generate an array representing the A/T/C/G counts for this nucleosome
                genomeData = line[52:]
                genomeData = genomeData.strip()
                genomeData = genomeData.split(",")
                genomeData = list(map(int, genomeData))

                # update the total A/T/C/G counts
                # this will be used to compute the average A/T/C/G counts after this loop
                avgA = avgA + genomeData[0]
                avgT = avgT + genomeData[1]
                avgC = avgC + genomeData[2]
                avgG = avgG + genomeData[3]

                # get portion of the line which has the nucleosome vector
                line = line[0:51]

                # add the genome data to the list of genome data entries
                observation_genome_data.append(genomeData)

                # add the observation to the list of observations
                observations.append(line.replace(",", ""))

            # if we've reached the end of the chromosome in the file
            if start and testChrom not in line and "chr" in line:# and "chr2" not in line:
                # break out the loop, we're done
                start = False
                
    # close the file
    f.close()

    # compute the average A/T/C/G counts for this chromosome
    avgA = avgA / len(observations)
    avgT = avgT / len(observations)
    avgC = avgC / len(observations)
    avgG = avgG / len(observations)


    NUMBER_OF_FOLDS = 10

    chunks = get_chunks(observations, int(len(observations)/NUMBER_OF_FOLDS), NUMBER_OF_FOLDS)
    genome_data_chunks = get_chunks(observation_genome_data, int(len(observation_genome_data)/NUMBER_OF_FOLDS), NUMBER_OF_FOLDS)

    final_test_index = cross_fold + 1
    if(final_test_index == len(chunks)):
      final_test_index = 0

    held_observations = chunks[cross_fold]
    held_observations_genome_data = genome_data_chunks[cross_fold]
    final_test_observations = chunks[final_test_index]
    final_test_observations_genome_data = genome_data_chunks[final_test_index]
    observations = []
    observation_genome_data = []

    for i in range(len(chunks)):
        if i != final_test_index:
            observations.extend(chunks[i])
            observation_genome_data.extend(genome_data_chunks[i])

    # S = the number of states
    S = state_count

    # if I've said the random flat when setting up this experiment, then I want to
    # generate random initial matricies, instead of using the ones we learned with the standard HMM
    if random:

        # get random values summing to 1 for each row, using the dirichlet distribution
        INITIAL_T = np.random.dirichlet(np.ones(S), 1)
        for i in range(S - 1):
            INITIAL_T = np.vstack([INITIAL_T, np.random.dirichlet(np.ones(S), 1)])

        # get random values summing to 1 for each row, using the dirichlet distribution
        INITIAL_E = np.random.dirichlet(np.ones(S), 1)
        for i in range(25):
            INITIAL_E = np.vstack([INITIAL_E, np.random.dirichlet(np.ones(S), 1)])

        # get random values summing to 1 for each row (in this case only 1 row), using 
        # the dirichlet distribution
        INITIAL_T0 = np.random.dirichlet(np.ones(S), 1)

    # if we don't want random initial matricies, then we use the ones learned in the standard HMM
    else:
      INITIAL_T, INITIAL_E, INITIAL_T0 = get_premade_matricies(S, cross_fold)


    representedHistones = {}
    representedHistones[12] = True
    representedHistones[8] = True
    representedHistones[3] = True
    representedHistones[17] = True

    # initialize the model, using the initial matricies generated above
    model = HiddenMarkovModel(INITIAL_T, INITIAL_E, INITIAL_T0, representedHistones, maxStep=50, epsilon = 0.001)

    model.final_test = final_test_observations
    model.final_test_data = final_test_observations_genome_data


    # calculate the time I start running
    start_time = datetime.datetime.now()

    # start the learning of the genomic weights, and save the learned matricies
    for c in range(len(chunks)):
      
      if c != final_test_index and c != cross_fold:
        chunk = chunks[c]
        data_chunk = genome_data_chunks[c]

        print("For sequence " + str(c) + " with test sequence " + str(final_test_index)+ " and with validation sequence " + str(cross_fold))

        model.learn_genomic_weights(chunk, data_chunk, avgA, avgT, avgC, avgG, held_observations, held_observations_genome_data, test_index)

    # print the learned genomic weights
    torch.set_printoptions(profile="full")
    print("Best Genomic Weights")
    print(model.genomic_weights)

    # calculate and print how much time it took to run
    total_time = datetime.datetime.now() - start_time
    print("total time: " + str(total_time))

    # add the total time and final log likelihood to formatted csv row 
    data_string = data_string + ", " + str(total_time) + ", " + str(model.scale_plot[-1].detach().numpy())

    # create a list representing the lengths of the various test sequences I want
    test_sequence_lengths = [5, 10, 20, 100, 0] # 0 for all

    # for each of these lengths
    for length in test_sequence_lengths:

        # initialize an empty test sequence
        test_seq = []

        # the 0 length is used to represent the whole 10% of the genome
        if length == 0:
            # so set the test_seq and the test_genome_data variables to be the
            # entire dataset held back from the training set
            test_seq = final_test_observations
            test_genome_data = final_test_observations_genome_data
        else:
            # otherwise, set the variables to a subset of the held back training data 
            test_seq = final_test_observations[0:length]
            test_genome_data = final_test_observations_genome_data[0:length]

        # print the length we're using
        print("test seq length: " + str(len(test_seq)))

        # get the log likelihood of the test sequence, given the releavant genome data
        held_back_likelihood = model.get_likelihood_of_seq(test_seq, test_genome_data)

        # print the log likelihood
        print("held_back_likelihood")
        print(held_back_likelihood)

        # add the log likelihood to the formatted string
        data_string = data_string + ", " + str(held_back_likelihood.detach())


    # use a very similar loop to generate some stats for these predictions
    # this very similar loop is separate because of how I like to format the data
    for length in test_sequence_lengths:

        # initialize an empty test sequence
        test_seq = []

        # the 0 length is used to represent the whole 10% of the genome
        if length == 0:
            # so set the test_seq and the test_genome_data variables to be the
            # entire dataset held back from the training set
            test_seq = final_test_observations
            test_genome_data = final_test_observations_genome_data
        else:
            # otherwise, set the variables to a subset of the held back training data 
            test_seq = final_test_observations[0:length]
            test_genome_data = final_test_observations_genome_data[0:length]

    # print the final data string
    data_string = data_string + "\n"
    print(data_string)

    # it its made it this far, then the experiment was successful
    return model.genomic_weights

# The following code actually starts off the experiments

final_weights = dict()

# run 1 tests
for t in range(1):
    for cross_fold in (range(10)):
        # for each state number
        for s in reversed(range(2, 11)):
            # print the chromosome, state #, and test #
            print("RUNNING EXPERIMENT: ")
            #print("chrom: " + chrom)
            print("S: " + (str(s)))
            print("Test Num: " + str(t))
            print("Cross Fold: " + str(cross_fold))

            success = run_experiment("chr2:", t, s, False, cross_fold)
            final_weights[str(s) + "_" + str(cross_fold)] = success

for key in final_weights:
  split = key.split("_")

  state = split[0]
  cf = split[1]

  print("if s==" + state +" and cross_fold==" + cf +":")
  print("\t" + final_weights[key])



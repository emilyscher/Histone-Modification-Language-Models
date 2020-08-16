import torch
import numpy as np
import pdb
import datetime
import torch.nn.functional as f
import matplotlib.pyplot as plt
import traceback
import gc
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn

# data structure for storing result lines
all_results = []

# lists for storing the matricies at various steps
ts_for_experiment = []
t0s_for_experiment = []
es_for_experiment = []

# dictionary containing the best matricies for each state number, 
# to be used in the custom HMMs
ts_for_custom = dict()
t0s_for_custom = dict()
es_for_custom = dict()

histones = ["H2AK5ac","H2AS129ph","H3K14ac","H3K18ac","H3K23ac","H3K27ac","H3K36me","H3K36me2","H3K36me3","H3K4ac","H3K4me","H3K4me2","H3K4me3","H3K56ac","H3K79me","H3K79me3","H3K9ac","H3S10ph","H4K12ac","H4K16ac","H4K20me","H4K5ac","H4K8ac","H4R3me","H4R3me2s","Htz1"]


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

    def __init__(self, T, E, T0, representedHistones, epsilon = 0.01, maxStep = 10):
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
        self.E = torch.tensor(E, dtype=torch.float64)
        # Transition matrix
        self.T = torch.tensor(T, dtype=torch.float64)
        # Initial state vector
        self.T0 = torch.tensor(T0, dtype=torch.float64)

        # list which will store the error at each step, to be used for creating plots
        self.scale_plot = list()

        # will hold the EM step number where convergenced was reached
        self.converged_at = 0

        # this line tells pytorch not to shorted printed matricies
        torch.set_printoptions(profile="full")

        # which histones will be represented in this experiment (useful for testing the predictive accuracy of one)
        self.representedHistones = representedHistones

    def initialize_viterbi_variables(self, shape): 
        pathStates = torch.zeros(shape, dtype=torch.float64)
        pathScores = torch.zeros_like(pathStates, dtype=torch.float64)
        states_seq = torch.zeros([shape[0]], dtype=torch.int64)
        return pathStates, pathScores, states_seq
    
    def belief_propagation(self, scores):
        return scores.view(-1,1) + torch.log(self.T)
    
    def viterbi_inference(self, x): # x: observing sequence        
        self.N = len(x)
        shape = [self.N, self.S]

        # Init_viterbi_variables
        pathStates, pathScores, states_seq = self.initialize_viterbi_variables(shape) 

        # log probability of emission sequence
        obs_prob_full = torch.log(self.generate_obs_prob_seq(x, -1))

        # initialize with state starting log-priors
        pathScores[0] = torch.log(self.T0) + obs_prob_full[0]
        for step, obs_prob in enumerate(obs_prob_full[1:]):
            # propagate state belief
            belief = self.belief_propagation(pathScores[step, :])
            # the inferred state by maximizing global function
            pathStates[step + 1] = torch.argmax(belief, 0)
            # and update state and score matrices 
            pathScores[step + 1] = torch.max(belief, 0)[0] + obs_prob


        # infer most likely last state
        states_seq[self.N - 1] = torch.argmax(pathScores[self.N-1, :], 0)
        for step in range(self.N - 1, 0, -1):
            # for every timestep retrieve inferred state
            state = states_seq[step]
            state_prob = pathStates[step][state]
            states_seq[step -1] = state_prob

        return states_seq, torch.exp(pathScores) # turn scores back to probabilities
        #return states_seq, pathScores # turn scores back to probabilities
    
    def initialize_forw_back_variables(self, shape):
        self.forward = torch.zeros(shape, dtype=torch.float64)
        self.backward = torch.zeros_like(self.forward, dtype=torch.float64)
        self.posterior = torch.zeros_like(self.forward, dtype=torch.float64)
        
    def _forward(model, obs_prob_seq, save_scale_plot):
        model.scale = torch.zeros([len(obs_prob_seq)], dtype=torch.float64) #scale factors
        # initialize with state starting priors
        init_prob = model.T0 * obs_prob_seq[0]
        # scaling factor at t=0
        model.scale[0] = 1.0 / init_prob.sum()

        # scaled belief at t=0
        model.forward[0] = model.scale[0] * init_prob

        # propagate belief
        for step, obs_prob in enumerate(obs_prob_seq[1:]):
            # previous state probability
            prev_prob = model.forward[step].unsqueeze(0)

            # transition prior
            prior_prob = torch.matmul(prev_prob, model.T)

            # forward belief propagation
            forward_score = prior_prob * obs_prob
            forward_prob = torch.squeeze(forward_score)

            # scaling factor
            model.scale[step + 1] = 1 / forward_prob.sum()
            # Update forward matrix
            model.forward[step + 1] = model.scale[step + 1] * forward_prob
    
    def _backward(self, obs_prob_seq_rev):

        # initialize with state ending priors
        self.backward[0] = self.scale[len(obs_prob_seq_rev) - 1] * torch.ones([self.S], dtype=torch.float64)

        # propagate belief
        for step, obs_prob in enumerate(obs_prob_seq_rev[:-1]):

            # next state probability
            next_prob = self.backward[step, :].unsqueeze(1)

            # observation emission probabilities
            obs_prob_d = torch.diag(obs_prob)
            # transition prior
            prior_prob = torch.matmul(self.T, obs_prob_d)
            # backward belief propagation
            backward_prob = torch.matmul(prior_prob, next_prob).squeeze()
            # Update backward matrix
            self.backward[step + 1] = self.scale[len(obs_prob_seq_rev) - 1 - step] * backward_prob

        # flipping self.backward
        self.backward = torch.flip(self.backward, [0])
        
    def forward_backward(self, obs_prob_seq, save_scale_plot):
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
        # call _forward on the observation seq
        self._forward(obs_prob_seq, save_scale_plot)
        # copy the observation seq and flip it
        obs_prob_seq_rev = torch.flip(obs_prob_seq, [0])
        
        # call _backward on the flipped observation seq
        self._backward(obs_prob_seq_rev)
    
        # these lines were added by me
        if(save_scale_plot):
            forward_prob = self.forward[-1]
            backward_prob = self.backward[-1] 

            product = torch.dot(forward_prob, backward_prob)
            
            otherLikelihood = -1 * torch.log(torch.sum(product)) * torch.sum(torch.log(self.scale))

            self.scale_plot.append(otherLikelihood)

            print("training seq likelihood = " + str(self.scale_plot[-1]))

            # check if the log likelihood has gotten worse, and if so print a message
            # as far as I can tell whenever this happens it has to do with underflow
            if len(self.scale_plot) > 1:
                if(self.scale_plot[-1] < self.scale_plot[-2]):
                    print("CURRENT TRAINING SEQUENCE LL IS WORSE THAN AT PREVIOUS EM STEP")

    # this method was added by me
    # it generates the emission matrix row for a particular observation
    # this is necessary because instead of having each vector combination 
    # being represented in the matricies, we're only having a parameter 
    # for each vector index
    def get_e_for_obs_num(self, obs):

        # initialize returned matrix
        returnE = torch.zeros(1, self.T.shape[0], dtype = torch.float64)

        # for each state
        for i in range(self.T.shape[0]):

            totprob = 1
            state = i

            # at the end of this loop, totprob will represent the probability 
            # of this emission vector being generated by state i
            for j in range(len(obs) - 1):
                mod = int(obs[j])

                place = j

                if mod == 1:
                    # the probability of the spot in the vector being 1 is given by self.E[place][state]
                    totprob = totprob * self.E[place][state]
                else:
                    # the probability of the spot in the vector being 0 is given by 1 - self.E[place][state]
                    totprob = totprob * (1 - self.E[place][state])

            returnE[0][i] = totprob

        return returnE

    def re_estimate_transition(self, x):
        self.M = torch.zeros([self.N - 1, self.S, self.S], dtype = torch.float64)

        for t in range(self.N - 1):

            tmp_0 = torch.matmul(self.forward[t].unsqueeze(0), self.T)

            # this line was changed by me, it was originally as follows:
            # tmp_1 = tmp_0 * self.E[x[t + 1]].unsqueeze(0)
            # this change allowed us to represent each histone modification in the 
            # emission matrix instead of each combination of mods
            tmp_1 = tmp_0 * self.get_e_for_obs_num(x[t + 1]).unsqueeze(0)
            
            denom = torch.matmul(tmp_1, self.backward[t + 1].unsqueeze(1)).squeeze()

            trans_re_estimate = torch.zeros([self.S, self.S], dtype = torch.float64)

            for i in range(self.S):
                numer = self.forward[t, i] * self.T[i, :] * self.get_e_for_obs_num(x[t+1]) * self.backward[t+1]
                trans_re_estimate[i] = numer / denom

            self.M[t] = trans_re_estimate

        self.gamma = self.M.sum(2).squeeze()

        T_new = self.M.sum(0) / self.gamma.sum(0).unsqueeze(1)

        T0_new = self.gamma[0,:]

        prod = (self.forward[self.N-1] * self.backward[self.N-1]).unsqueeze(0)
        s = prod / prod.sum()
        self.gamma = torch.cat([self.gamma, s], 0)
        self.prob_state_1.append(self.gamma[:, 0]) 
        return T0_new, T_new
    
    def re_estimate_emission(self, x):

        states_marginal = self.gamma.sum(0)

        # One hot encoding buffer that you create out of the loop and just keep reusing
        seq_one_hot = torch.zeros([len(x), self.O], dtype=torch.float64)

        # these lines were added by me
        # instead of creating a one hot vector representing which emission is 
        # present at each point in the observation sequence, I'm creating a 
        # vector representing all of the histone modifications which are 
        # present in the emission
        #
        # an example of line from the previous version of this bit would be:
        # 0, 0, 0, ..., 1, ,,,, 0, where the number of entries is equal to the 
        # number of histone modification combinations seen in the whole observation seqeunce
        #
        # now the output looks something like this:
        # 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0
        # where each entry represents the precense/absence of a particular modification
        for i in range(len(x)):
            for j in range(len(x[i])):
                seq_one_hot[i][j] = int(x[i][j])

        emission_score = torch.matmul(seq_one_hot.transpose_(1, 0), self.gamma)

        return emission_score / states_marginal
    
    def check_convergence(self, new_T0, new_transition, new_emission):
  
        delta_T0 = torch.max(torch.abs(self.T0 - new_T0)).item() < self.epsilon
        delta_T = torch.max(torch.abs(self.T - new_transition)).item() < self.epsilon
        delta_E = torch.max(torch.abs(self.E - new_emission)).item() < self.epsilon

        print("delta_T0 = " + str(torch.max(torch.abs(self.T0 - new_T0)).item()))
        print("delta_T = " + str(torch.max(torch.abs(self.T - new_transition)).item()))
        print("delta_E = " + str(torch.max(torch.abs(self.E - new_emission)).item()))

        return delta_T0 and delta_T and delta_E
    
    # this method was added by me
    # this generates a matrix which represents the likelihoods a particular 
    # emission in the observation sequence came from each state
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
                    obs_prob_seq[x][y] = 0.000000000000000000001

        return obs_prob_seq

    def expectation_maximization_step(self, obs_seq):

        # probability of emission sequence
        # this line is new. It was previous as follows:
        # obs_prob_seq = self.E[obs_seq]
        # this change allowed us to represent each histone modification in the 
        # emission matrix instead of each combination of mods
        obs_prob_seq = self.generate_obs_prob_seq(obs_seq, -1)

        # run forward_backward
        self.forward_backward(obs_prob_seq, True)

        # re-estimate the transition and T0 matricies
        new_T0, new_transition = self.re_estimate_transition(obs_seq)

        # re-estimate the emission matrix
        new_emission = self.re_estimate_emission(obs_seq)

        # check for convergence
        converged = self.check_convergence(new_T0, new_transition, new_emission)

        print("E STEP COMPLETED, UPDATING MATRICIES")

        # update matricies based on new estimations
        new_T0 = new_T0
        self.T0 = new_T0
        self.E = new_emission
        self.T = new_transition
        
        return converged

    # this method was added by me
    # it calculates the log likelihood of a sequence by running 
    # forward_backward once
    def get_likelihood_of_seq(self, obs_seq):
       
        torch.set_printoptions(profile="full")

        # probability of emission sequence
        obs_prob_seq = self.generate_obs_prob_seq(obs_seq, -1)

        # shape of Variables
        shape = [len(obs_seq), self.S]

        # initialize variables
        self.initialize_forw_back_variables(shape)

        # run forward backward once
        self.forward_backward(obs_prob_seq, False)

        forward_prob = self.forward[-1]
        backward_prob = self.backward[-1]

        product = torch.dot(forward_prob, backward_prob)

        otherLikelihood = -1 * torch.log(torch.sum(product)) * torch.sum(torch.log(self.scale))

        if(otherLikelihood != otherLikelihood):
            print(self.scale)
            print(self.forward)
            print(self.backward)

            print(self.forward.size())

            print(0/0)

        # reset variables
        shape = [self.N, self.S]
        self.initialize_forw_back_variables(shape)

        return otherLikelihood

    def posterior_decoding(self, seq, index_to_ignore):
        # probability of emission sequence
        obs_prob_seq = self.generate_obs_prob_seq(seq, index_to_ignore)

        # shape of Variables
        shape = [len(seq), self.S]

        # initialize variables
        self.initialize_forw_back_variables(shape)

        # run forward backward once
        self.forward_backward(obs_prob_seq, False)

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

    def getStatsForLogReg(self, observations, test_seq):
        accuracies = {}
        precisions = {}
        recalls = {}

        for test_index in range(len(observations[0])):
            if test_index in self.representedHistones.keys():
                guessed_sequence = self.guess_sequence_logreg(observations, test_seq, 0, test_index)
                accuracy, precision, recall = generate_stats(guessed_sequence, test_seq, test_index)
                accuracies[test_index] = accuracy
                precisions[test_index] = precision
                recalls[test_index] = recall

        return accuracies, precisions, recalls

    def guess_sequence_logreg(self, training_sequence, test_sequence, pred_type, histone_index_to_test):

        new_test_seq = test_sequence.copy()

        for i in range(len(new_test_seq)):
            new_test_seq[i] = new_test_seq[i][:histone_index_to_test] + new_test_seq[i][(histone_index_to_test + 1):]

            

        # now we want to use logistic regression to predict one histone modification for the test sequence
        # pred_type=1 -> only ATCG
        # pred_type=2 -> only state
        # pred_type=3 -> both

        test_state_seq = self.posterior_decoding(new_test_seq, histone_index_to_test)
        train_state_seq = self.posterior_decoding(training_sequence, -1)

        final_seq = []

        logregdata = {}
        logregdata['state'] = []
        logregdata['emission_prob'] = []

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

            if not empty or emptyCount < 5:
                for i in range(len(obs)):
                    if str(i) not in logregdata:
                        logregdata[str(i)] = []

                    logregdata[str(i)].append(int(obs[i]))
                logregdata['state'].append(train_state_seq[x])
                logregdata['emission_prob'].append(self.E[histone_index_to_test][train_state_seq[x]])

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
            prediction_data['state'] = [i]
            prediction_data['emission_prob'] = [self.E[histone_index_to_test][i]]
            for h in range(len(full_vector)):
                if h != histone_index_to_test:
                    prediction_data[str(h)] = [int(full_vector[h])]

            X_test = pd.DataFrame(prediction_data,columns= prediction_data.keys())
            y_pred=logistic_regression.predict(X_test)

            char = str(y_pred[0])

            vector = vector[:histone_index_to_test] + char + vector[histone_index_to_test:]

            final_seq.append(vector)

        return final_seq

    # predict a sequence of a particular @length based on trained HMM and genomic weights
    # always chooses most likely emission/transition
    # this method is new, and is the same as in the CustomHMM
    def guess_sequence_max(self, length, training_seq):

        # initialize list of predicted emissions
        emission_list = []

        # get the most likely state sequence for the training data
        state_seq, path_scores = self.viterbi_inference(training_seq)

        # get the last state in the sequence
        state = state_seq[-1]

        # get the relevant portion of the emission sequence based on the state
        relevantEmission = self.E[:, state:(state + 1)]

        # initialize emission
        emission = ""

        # for each histone modification
        for e in relevantEmission:

            # if value in the emission matrix is greater than 0.5, then the 
            # value for that spot should be 1, otherwise 0
            if e <= 0.5:
                emission = emission + "0"
            else:
                emission = emission + "1"

        # add predicted vector to the predicted emission sequence
        emission_list.append(emission)

        # do the same as above for each predicted state
        for i in range(length - 1):

            # get the probabilities for each state transition given the current state
            possibleTransitions = self.T[state]

            # choose the most likely state
            state = self.get_max_index(possibleTransitions)

            # get the relevant part of the emission matrix for that state
            relevantEmission = self.E[:, state:(state + 1)]

            # initialize emission
            emission = ""

            # for each histone modification
            for e in relevantEmission:

                # if value in the emission matrix is greater than 0.5, then 
                # the value for that spot should be 1, otherwise 0
                if e <= 0.5:
                    emission = emission + "0"
                else:
                    emission = emission + "1"

            # add predicted vector to the predicted emission sequence
            emission_list.append(emission)

        # return predicted sequence of vectors
        return emission_list

    # predict a sequence of a particular @length based on trained HMM and genomic weights
    # makes biased random guesses to predict transitions and emissions, based on the probabilities 
    # in the learning matricies
    # this method is new
    def guess_sequence_prob(self, length, training_seq):

        # initialize list of predicted emissions
        emission_list = []

        # get the most likely state sequence for the training data
        state_seq, path_scores = self.viterbi_inference(training_seq)

        # get the last state in the sequence
        state = state_seq[-1]

        # get the relevant portion of the emission sequence based on the state
        relevantEmission = self.E[:, state:(state + 1)]

        # initialize emission
        emission = ""

        # for each histone modification
        for e in relevantEmission:
            # guess whether the vector value should be 1 or 0 based on the emission matrix entry
            guess = self.guess_emission(e)

            if guess == 0:
                emission = emission + "0"
            else:
                emission = emission + "1"

        # add predicted vector to the predicted emission sequence
        emission_list.append(emission)

        # do the same as above for each predicted state
        for i in range(length - 1):

            # get the probabilities for each state transition given the current state
            possibleTransitions = self.T[state]

            # guess which state to transition to, based on the adjusted transition matrix probabilities
            state = self.get_guess(possibleTransitions)

            # get the relevant part of the emission matrix for that state
            relevantEmission = self.E[:, state:(state + 1)]

            # initialize emission
            emission = ""

            # for each histone modification
            for e in relevantEmission:
                # guess whether the vector value should be 1 or 0 based on the emission matrix entry
                guess = self.guess_emission(e)

                if guess == 0:
                    emission = emission + "0"
                else:
                    emission = emission + "1"

            # add predicted vector to the predicted emission sequence
            emission_list.append(emission)

        # return predicted sequence of vectors
        return emission_list

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
    def save_figure(self, chromosome, test_num, cross_fold):

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
        plt.savefig("./testing_likelihood_at_each_step/5_2_20_standard_chrom_" + chrom_name + "_states_" + str(self.S) + "_testnum_" + str(test_num) + "_crossfold_" + str(cross_fold) +"_allhistonestested.png")

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
        plt.savefig("./testing_likelihood_at_each_step_likelihoods/5_2_20_standard_chrom_" + chrom_name + "_states_" + str(self.S) + "_testnum_" + str(test_num) + "_crossfold_" + str(cross_fold) +"_allhistonestested.png")

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
        plt.savefig("./testing_likelihood_at_each_step_"+name+"/5_2_20_standard_chrom_" + chrom_name + "_states_" + str(self.S) + "_testnum_" + str(test_num) + "_crossfold_" + str(cross_fold) +"_allhistonestested.png", bbox_inches="tight")

        # start a new plot
        plt.clf()



    def Baum_Welch_EM(self, obs_seq, test_seq, histone_to_test):
        self.test_index = histone_to_test

        # length of observed sequence
        self.N = len(obs_seq)

        # shape of Variables
        shape = [self.N, self.S]

        # initialize variables
        self.initialize_forw_back_variables(shape)

        converged = False

        # initialize the list of validation sequence likelihoods at each step
        self.test_seq_likelihoods = []

        # initialize the variables which will hold the matricies from previous EM steps
        self.previous_transitions = []
        self.previous_emissions = []
        self.previous_t0s = []

        initialLikelihood = self.get_likelihood_of_seq(obs_seq)
        held_back_likelihood = self.get_likelihood_of_seq(test_seq)

        self.accuracies = {}
        self.precisions = {}
        self.recalls = {}
        # clear scale plot, because we don't want this extra likelihood calculation included
        self.scale_plot = []

        print("training seq log likelihood before first descent step = " + str(initialLikelihood))
        print("test seq log likelihood before first descent step = " + str(held_back_likelihood))

        for i in range(self.maxStep):

            print("step " + str(i) +" at " +str(datetime.datetime.now()), flush=True)

            # run EM step
            converged = self.expectation_maximization_step(obs_seq)

            # save the old matricies
            self.previous_transitions.append(self.T.clone())
            self.previous_emissions.append(self.E.clone())
            self.previous_t0s.append(self.T0.clone())

            # generate the likelihood of the test sequence based on this step's matricies
            held_back_likelihood = self.get_likelihood_of_seq(test_seq)

            # print test sequence likelihood
            print("validation seq log likelihood = " + str(held_back_likelihood))

            # add the test sequence's likelihood to the list of test sequence likelihoods, so we can plot later
            self.test_seq_likelihoods.append(held_back_likelihood)

            if(len(self.test_seq_likelihoods) > 1 and self.test_seq_likelihoods[-1] < self.test_seq_likelihoods[-2]):
                print("CURRENT VALIDATION SEQUENCE LL IS WORSE THAN @ PREVIOUS DESCENT STEP")

            if converged:
                print('converged at step {}'.format(i))
                self.converged_at = i

                # print the likelihoods, in case I want to plot again later
                print("scale plot")
                print(self.scale_plot)
                print("test seq likelihoods plot")
                print(self.test_seq_likelihoods)

                # break out of the loop
                break 
     
        # this is new
        # figure out which matricies were assosciated with the best validation sequence likelihood
        maxIndex = -1
        maxLikelihood = -1 * np.inf
        for x in range(len(self.test_seq_likelihoods)):
            if (self.test_seq_likelihoods[x] > maxLikelihood):
                maxLikelihood = self.test_seq_likelihoods[x]
                maxIndex = x

        print("Max test seq likelihood= " + str(maxLikelihood))
        print("Max test seq likelihood index= " + str(maxIndex))

        print("Max test seq likelihood matricies: ")
        print("transition")
        print(self.previous_transitions[maxIndex])
        print("emission")
        print(self.previous_emissions[maxIndex])
        print("t0")
        print(self.previous_t0s[maxIndex])


        # set the model parameters to the best versions (with respect to the 
        # validation sequence), so that the stats gathered are based on that
        self.T0 = self.previous_t0s[maxIndex]
        self.T = self.previous_transitions[maxIndex]
        self.E = self.previous_emissions[maxIndex]

        # return the matricies
        return self.previous_t0s[maxIndex], self.previous_transitions[maxIndex], self.previous_emissions[maxIndex], self.scale_plot[maxIndex], converged

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
        precision = np.nan

    # compute recall (if possible)
    if (truePositives + falseNegatives) != 0:
        recall = truePositives / (truePositives + falseNegatives)
    else:
        recall = np.nan

    # compute f1 (if possible)
    if (precision + recall) != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = np.nan

    # compute accuracy (if possible)
    accuracy = (truePositives + trueNegatives) / (trueNegatives + truePositives + falsePositives + falseNegatives)

    # format the return string
    retString = retString + ", " + str(trueNegatives) + ", " + str(truePositives) + ", " + str(falseNegatives) + ", " + str(falsePositives) + ", " + str(precision) + ", " + str(recall) + ", " + str(f1) + ", " + str(accuracy) + ", " + str((trueNegatives + truePositives + falsePositives + falseNegatives))

    return accuracy, precision, recall


def get_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# method to make running a bunch of experiments easier
# takes a chromosome, test ID number, state count
# this method is new
def run_experiment(testChrom, test_num, state_count, cross_fold, INITIAL_T, INITIAL_E, INITIAL_T0):

    # initialize the string which will eventually be a formatted data row in a CSV file
    data_string = testChrom[0:-1] + ", " + str(state_count) + ", " + str(test_num) + ", " + str(cross_fold) 

    # initialize list which will be the list of observations in the training set
    observations = []

    # initialize observation count
    count = 0

    # flag for determining if we've reached the correct point in the input file for the chromosome
    start = True

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

                # get portion of the line which has the nucleosome vector
                line = line[0:51]

                # add the observation to the list of observations
                observations.append(line.replace(",", ""))

            # if we've reached the end of the chromosome in the file
            if start and testChrom not in line and "chr" in line:
                # break out of the loop, we're done
                start = False
          
    # close the file
    f.close()

    # is actually 10 because the test seq/validation seq are flipped half the time
    NUMBER_OF_FOLDS = 5

    half_string = "fullchromosome"

    flip_test_seq = False

    adjusted_cross_fold = cross_fold

    if cross_fold >= 5:
        flip_test_seq = True
        adjusted_cross_fold = adjusted_cross_fold - 5


    chunks = list(get_chunks(observations, int(len(observations)/NUMBER_OF_FOLDS)))

    while len(chunks) > NUMBER_OF_FOLDS:
        chunks[-2] = chunks[-2] + chunks[-1]
        del chunks[-1]

    held_observations = chunks[adjusted_cross_fold]
    final_test_observations = []
    observations = []

    for i in range(len(chunks)):
        if i != adjusted_cross_fold:
            observations.extend(chunks[i])

    if not flip_test_seq:
        final_test_observations = held_observations[len(held_observations)//2:]
        held_observations = held_observations[:len(held_observations)//2]
    else:
        final_test_observations = held_observations[:len(held_observations)//2]
        held_observations = held_observations[len(held_observations)//2:]

    representedHistones = {}
    representedHistones[12] = True
    representedHistones[8] = True
    representedHistones[3] = True
    representedHistones[17] = True

    # S = the number of states
    S = state_count

    # initialize the model, using the initial matricies generated above
    model = HiddenMarkovModel(INITIAL_T, INITIAL_E, INITIAL_T0, representedHistones, maxStep=200, epsilon = 0.1)

    # calculate the time I start running
    start_time = datetime.datetime.now()

    test_index = 12

    # start the learning of the genomic weights, and save the learned matricies
    trans0, transition, emission, final_likelihood, c = model.Baum_Welch_EM(observations, held_observations, test_index)
    
    # print the initial and final matricies, in case I need them later or want to test something
    torch.set_printoptions(profile="full")
    print("Initial Matricies")
    print("INITIAL_T")
    print(INITIAL_T)
    print("INITIAL_E")
    print(INITIAL_E)
    print("INITIAL_T0")
    print(INITIAL_T0)

    
    print("Final Matricies")
    print("transition")
    print(transition)
    print("emission")
    print(emission)
    print("trans0")
    print(trans0)

    # add the final matricies to a list of all of the best maticires for this state number
    ts_for_experiment.append(transition)
    t0s_for_experiment.append(trans0)
    es_for_experiment.append(emission)

    # calculate and print the total time it took to run
    total_time = datetime.datetime.now() - start_time
    print("total time: " + str(total_time))

    # add the total time and final training seq log likelihood to the data string
    data_string = data_string + ", " + str(total_time) + ", " + str(final_likelihood)

    # create a list representing the lengths of the various test sequences I want
    test_sequence_lengths = [5, 10, 20, 100, 0] # 0 for all

    # dictionary represnting the testing sequence log likelihoods for a test sequence length
    heldBackLikelihoods = dict()

    # for each of these lengths
    for length in test_sequence_lengths:
        # initialize an empty test sequence
        test_seq = []

        # the 0 length is used to represent the whole 10% of the genome
        if length == 0:
            # so set the test_seq and the test_genome_data variables to be the
            # entire dataset held back from the training set
            test_seq = final_test_observations
        else:
            # otherwise, set the variables to a subset of the held back training data 
            test_seq = final_test_observations[0:length]

        # print the length we're using
        print("test seq length: " + str(len(test_seq)))

        # get the log likelihood of the test sequence, given the releavant genome data
        held_back_likelihood = model.get_likelihood_of_seq(test_seq)

        # print the log likelihood
        print("final test observations held_back_likelihood")
        print(held_back_likelihood)

        # add the LL to the data string
        data_string = data_string + ", " + str(held_back_likelihood.detach())

        # add the LL for the test sequence length to the dict
        heldBackLikelihoods[length] = held_back_likelihood

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
        else:
            # otherwise, set the variables to a subset of the held back training data 
            test_seq = final_test_observations[0:length]

        # predict an emission sequence, using the guess_sequence_max method, and print the assosciated stats
        print("Logreg Guess Statistics:")

    data_string = data_string + "\n"
    all_results.append(data_string)

    # print the final data string
    print(data_string)

    # return the LL of the longest test seq
    return heldBackLikelihoods[0], model.T, model.E, model.T0


# there are 16 chromosomes, but for now I'm only testing one
# colon included because this is a lazy way of doing it :(
chromosomes = ["chr1:"]

transitions = dict()
emissions = dict()
inittrans = dict()

for s in range(2, 11):
    for t in range(1):
        INITIAL_T = np.random.dirichlet(np.ones(s), 1)
        for i in range(s - 1):
            INITIAL_T = np.vstack([INITIAL_T, np.random.dirichlet(np.ones(s), 1)])

        # Initialize E with random values summing to 1 for each row, using the dirichlet distribution
        INITIAL_E = np.random.dirichlet(np.ones(s), 1)
        for i in range((26) - 1):
            INITIAL_E = np.vstack([INITIAL_E, np.random.dirichlet(np.ones(s), 1)])

        # Initialize T0 with random values summing to 1 for each row (in this case only 1 row), using 
        # the dirichlet distribution
        INITIAL_T0 = np.random.dirichlet(np.ones(s), 1)

        key = str(s) + "_" + str(t)

        transitions[key] = INITIAL_T
        emissions[key] = INITIAL_E
        inittrans[key] = INITIAL_T0

NUMBER_OF_CROSS_FOLDS = 10

bestLikelihoods = dict()

bestMatrixStrings = []

for chrom in chromosomes:
    for s in reversed(range(2, 11)):
        for cross_fold in range(NUMBER_OF_CROSS_FOLDS):

            bestLikelihoods[str(s)+"_"+str(cross_fold)] = -10000000000
            bestLikelihoodT = []
            bestLikelihoodE = []
            bestLikelihoodT0 = []

            for t in range(1):
                
                key = str(s) + "_" + str(t)
                INITIAL_T = transitions[key] 
                INITIAL_E = emissions[key]
                INITIAL_T0 = inittrans[key]

                print("RUNNING EXPERIMENT: ")
                print("chrom: " + chrom)
                print("S: " + (str(s)))
                print("Test Num: " + str(t))
                print("Cross Fold: " + str(cross_fold))

                likelihood_of_test_seq, T, E, T0 = run_experiment(chrom, t, s, cross_fold, INITIAL_T, INITIAL_E, INITIAL_T0)

                #if likelihood_of_test_seq > bestLikelihoods[str(s)+"_"+str(cross_fold)]:
                bestLikelihoods[str(s)+"_"+str(cross_fold)] = likelihood_of_test_seq
                bestLikelihoodT = T
                bestLikelihoodE = E
                bestLikelihoodT0 = T0


            matrixString = "\tif s == " + str(s) + " and cross_fold == "+str(cross_fold)+":\n" + "\t\tINITIAL_T = torch." + str(bestLikelihoodT) + "\n" + "\t\tINITIAL_E = torch." + str(bestLikelihoodE) + "\n" + "\t\tINITIAL_T0 = torch." + str(bestLikelihoodT0) + "\n"

            bestMatrixStrings.append(matrixString)


for s in bestMatrixStrings:
    print(s)

print("\t return INITIAL_T, INITIAL_E, INITIAL_T0\n\n")

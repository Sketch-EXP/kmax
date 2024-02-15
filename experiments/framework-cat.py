import numpy as np
from numpy import random
import random
from numpy.random import uniform, choice
import math
np.random.seed(90)
random.seed(2019)

def generate_data(rounds, probs, vals):
    dataset = np.c_[np.random.choice(a = vals[0], size = rounds, p = probs[0])]
    for i in range(1,len(probs)):
        data = np.c_[np.random.choice(a = vals[i], size = rounds, p = probs[i])]
        dataset = np.hstack([dataset,data] )
    return dataset


## Baseline MAB class
class baseUCB():
    def __init__(self, number_of_rounds, narms, rewards):
        self.narms = narms
        self.T = np.zeros((number_of_rounds, narms))
        self.mean = np.zeros((number_of_rounds, narms))
        self.U = np.zeros((number_of_rounds, narms))
        self.q0 = np.ones(narms)*np.inf
        self.S = np.zeros(number_of_rounds, dtype=np.int32) # selected arm
        self.val = np.zeros(number_of_rounds)  # winner value
        self.regrets = np.zeros(number_of_rounds)
        self.best_r = max(rewards)

    def off_max(self, val, q0):
        ## oracle selection
        if len(np.where(q0==0)[0])<84:
            # choose an arm with maximum q0
            action = np.random.choice(np.where(q0==max(q0))[0])
            # after the arm is chosen, set the corresponding Q0 value to zero
            self.q0[action]=0
        else:
        # Now, after that we ensure that there is no np.inf in Q0 values and all of them are set to zero
        # we return to play based on average mean rewards
            action = np.random.choice(np.where(val==max(val))[0])
        return action


    def update_ucb(self, i, t):
        if self.T[t-1, i] > 0:
            rho = np.sqrt((1.5*np.log(t))/self.T[t-1, i])
            return self.mean[t-1, i] + rho
        else:
            return self.mean[t-1, i]

    def update_est(self, t):
        self.T[t] = self.T[t-1]
        self.mean[t] = self.mean[t-1]
        self.T[t, self.S[t]] += 1
        self.mean[t, self.S[t]] = (self.T[t,self.S[t]] - 1) / float(self.T[t,self.S[t]]) * self.mean[t, self.S[t]]  + (1 / float(self.T[t, self.S[t]]))* self.val[t]

    def ucb_round(self, t, data, sets, rewards):
        self.U[t] = [self.update_ucb(i, t) for i in range(self.narms)]  # update UCBs
        self.S[t] = self.off_max(self.U[t], self.q0)  # feed UCBs to offline oracles
        # arm reward
        curr_data = data[t-1]
        self.val[t] = np.max(curr_data[list(sets[self.S[t]])])

        reg = self.best_r - rewards[self.S[t]]
        self.regrets[t] = np.abs(reg)
        self.update_est(t)

        
        
# Modified CUCB class
## Modified CUCB algorithm
class modiCUCB():
    def __init__(self, number_of_rounds, N, C, K):  # add upper bound on support size
        self.rounds = number_of_rounds
        self.N = N
        self.K = K
        self.T = np.zeros((number_of_rounds, N, C+1))
        self.U = np.zeros((number_of_rounds, N, C+1))  # UCBs
        self.p = np.zeros((number_of_rounds, N, C+1))
        self.v = np.zeros((N, C+1))
        self.sigma = np.zeros(N, dtype=np.int32)  ## counter
        self.p[0,:,:] = 1
        self.v[:] = 1   # initializations
        self.S = np.zeros((number_of_rounds, K), dtype=np.int32) # selected set
        self.ind = np.zeros(number_of_rounds, dtype=np.int32) # winner index
        self.val = np.zeros(number_of_rounds)  # winner value
        self.regrets = np.zeros(number_of_rounds)
        
    def transform_binary(self,probs):
        res = []
        for p in probs:
            pt = [p[len(p)-1]]*len(p)
            for i in range(len(p)-1):
                pt[i] = round(p[i]/(1-sum(p[i+1:])),2)
            res.append(pt)
        return res

    ## calculate reward
    def reward(self, s, p, v):
        s = np.sort(s)[::-1]
        p = np.array([p[i] for i in s]).flatten()
        v = np.array([v[i] for i in s]).flatten()
        index = np.argsort(v)[::-1]
        r = p[index[0]] * v[index[0]]
        fac = 1
        for k in range(1,len(index)):
            fac = fac * (1 - p[index[k-1]])
            r += fac * p[index[k]] * v[index[k]]
        return r

    ## calculate best reward
    def best_r(self, probs, vals):
        p = self.transform_binary(probs)
        best_r = self.reward([8,7,6], p, vals)
        return best_r

    ## update ucbs
    ## update for those index in counters
    def update_ucb(self, i, t):
        res = np.zeros(C+1)
        for j in range(self.sigma[i]+1):
            if self.T[t-1, i, j] > 0:
                rho = np.sqrt((1.5*np.log(t))/self.T[t-1, i, j])
                res[j] = min(self.p[t-1, i, j] + rho, 1)
            else:
                res[j] = 1
        return res

    ## offline oracle
    def off_kmax(self, probs, vals):
        S = []
        unvisited = [k for k in range(self.N)]
        means = [self.reward([i], probs, vals) for i in range(self.N)]
        item = np.argmax(means)
        S.append(item)
        unvisited.remove(item)
        
        for i in range(1, self.K):
            r = np.zeros(self.N)
            for item in unvisited:
                nxt_S = np.append(S, item)
                r[item] = self.reward(nxt_S, probs, vals)
            if np.all(r[r != 0] == r[0]):
                nxt = random.choice(unvisited)
            else:
                nxt = np.argmax(r)
            S.append(nxt)
            unvisited.remove(nxt)
        return S

    ## update estimates
    ### extra care for items with known values
    def update_est(self, t):
        self.T[t] = self.T[t-1]
        self.p[t] = self.p[t-1]
        # updates for the winner
        if self.ind[t] < self.N:
            if self.val[t] not in self.v[self.ind[t]]:
                self.sigma[self.ind[t]] += 1  ## increase counter 
                self.v[self.ind[t], self.sigma[self.ind[t]]] = self.val[t] 
                self.T[t, self.ind[t], self.sigma[self.ind[t]]] = 1  # reset to one
                self.p[t, self.ind[t], self.sigma[self.ind[t]]] = 1       
            else:
                j = np.where(self.v[self.ind[t]] == self.val[t])[0][0]
                self.T[t, self.ind[t], j] += 1
                self.p[t, self.ind[t], j]  = 1/self.T[t, self.ind[t], j] + (1-1/self.T[t, self.ind[t], j])*self.p[t-1, self.ind[t], j]
        
        for k in range(self.K):
            ## For indices, we should only update those inside items selected!
            for j in range(int(self.sigma[self.S[t, k]]+1)):
                if self.v[self.S[t, k], j] > self.val[t]:   # updates for those observed
                    self.T[t, self.S[t, k], j] += 1
                    self.p[t, self.S[t, k], j] = (1-1/self.T[t, self.S[t, k], j])*self.p[t-1, self.S[t, k], j]


    ## one round of algorithm
    def kmax_round(self, t, data, probs, vals):
        self.U[t] = [self.update_ucb(i, t) for i in range(self.N)]  # update UCBs
        self.S[t] = self.off_kmax(self.U[t],self.v)  # feed UCBs to offline oracles
        # item reward
        curr_data = data[t-1][self.S[t]]
        reg = self.best_r(probs, vals) - self.reward(self.S[t], self.transform_binary(probs), vals)
        self.regrets[t] = np.abs(reg)
        # index of winner
        if np.sum(curr_data) > 0:
            self.ind[t] = int(self.S[t][np.argmax(curr_data)])
            self.val[t] = np.max(curr_data)
        else:
            self.ind[t] = self.N
            self.val[t] = 0
        self.update_est(t)
        
        
class semiCUCB():
    def __init__(self, number_of_rounds, N, C, K):
        self.rounds = number_of_rounds
        self.N = N
        self.K = K
        self.T = np.zeros((number_of_rounds, N, C))
        self.U = np.zeros((number_of_rounds, N, C))  # UCBs
        self.p = np.zeros((number_of_rounds, N, C))
        self.v = np.zeros((N, C))
        self.p[0,:,:] = 1
        self.v[:] = 1   # initializations
        self.S = np.zeros((number_of_rounds, K), dtype=np.int32) # selected set
        self.regrets = np.zeros(number_of_rounds)
        
    ## calculate reward
    def reward(self,s, p, v):
        s = np.sort(s)[::-1]
        p = np.array([p[i] for i in s]).flatten()
        v = np.array([v[i] for i in s]).flatten()
        index = np.argsort(v)[::-1]
        r = p[index[0]] * v[index[0]]
        fac = 1
        for k in range(1,len(index)):
            fac = fac * (1 - p[index[k-1]])
            r += fac * p[index[k]] * v[index[k]]
        return r
    
    def transform_binary(self,probs):
        res = []
        for p in probs:
            pt = [p[len(p)-1]]*len(p)
            for i in range(len(p)-1):
                pt[i] = round(p[i]/(1-sum(p[i+1:])),2)
            res.append(pt)
        return res
    
    ## calculate best reward
    def best_r(self, probs, vals):
        p = self.transform_binary(probs)
        best_r = self.reward([8,7,6], p, vals)
        return best_r
    
    ## update ucbs
    ## for all arms that are not observed, we use fictitious arms to hold
    def update_ucb(self, i, t):
        res = np.zeros(C)
        for j in range(C):
            if self.T[t-1, i, j] > 0:
                rho = np.sqrt((1.5*np.log(t))/self.T[t-1, i, j])
                res[j] = min(self.p[t-1, i, j] + rho, 1)
            else:
                res[j] = 1
        return res

    
    ## offline oracle
    def off_kmax(self, probs, vals):
        S = []
        unvisited = [k for k in range(self.N)]
        means = [self.reward([i], probs, vals) for i in range(self.N)]
        item = np.argmax(means)
        S.append(item)
        unvisited.remove(item)
        
        for i in range(1, self.K):
            r = np.zeros(self.N)
            for item in unvisited:
                nxt_S = np.append(S, item)
                r[item] = self.reward(nxt_S, probs, vals)
            if np.all(r[r != 0] == r[0]):
                nxt = random.choice(unvisited)
            else:
                nxt = np.argmax(r)
            S.append(nxt)
            unvisited.remove(nxt)
        return S

    
    def update_est_semi(self, t, data):
        self.T[t] = self.T[t-1]
        self.p[t] = self.p[t-1]
        for k in range(self.K):
            if data[self.S[t, k]] not in self.v[self.S[t, k]]:
                j = np.where(self.v[self.S[t, k]] == 1)[0][0] ## new value
                self.v[self.S[t, k], j] = data[self.S[t, k]]
                self.T[t, self.S[t, k], j] = 0
            j = np.where(self.v[self.S[t, k]] == data[self.S[t, k]])[0][0]
            self.T[t, self.S[t, k], j] += 1
            self.p[t, self.S[t, k],j]  = 1/self.T[t, self.S[t, k],j] + (1-1/self.T[t, self.S[t, k],j])*self.p[t-1, self.S[t, k],j]    
            for j in range(C):
                if self.v[self.S[t, k], j] > data[self.S[t, k]]:   ## for other values of this item
                    self.T[t, self.S[t, k], j] += 1
                    self.p[t, self.S[t, k], j] = (1-1/self.T[t, self.S[t, k], j])*self.p[t-1, self.S[t, k], j]

            
    def kmax_round_semi(self, t, data, probs, vals):
        self.U[t] = [self.update_ucb(i, t) for i in range(self.N)]  # update UCBs
        self.S[t] = self.off_kmax(self.U[t],self.v)  # feed UCBs to offline oracles
        # item reward for each in the set
        curr_data = data[t-1][self.S[t]]
        reg = self.best_r(probs, vals) - self.reward(self.S[t], self.transform_binary(probs), vals)
        self.regrets[t] = np.abs(reg)
        self.update_est_semi(t, data[t-1])
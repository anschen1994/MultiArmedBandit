import numpy as np 



class Policy():
    def __init__(self, **kwargs):
        pass

    def initialize(self):
        raise NotImplementedError

    def update_memory(self, i, reward):
        if reward == 1:
            self.memory[i][0] += 1
        else:
            self.memory[i][1] += 1

    def choose_action(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def run(self, num_steps):
        self.initialize()
        cum_reward = []
        for i in range(num_steps):
            cum_reward.append(self.step())
        return cum_reward




class GreedyPolicy(Policy):
    def __init__(self, memory=None, estimators=None, bandit=None, eps=0.0):
        self.bandit = bandit
        if memory is None:
            self.memory = dict([(i, [0,0]) for i in range(self.bandit.n)])
        else:
            self.memory = memory
        self.eps = eps
        if estimators is None:
            self.estimators = [0.5] * self.bandit.n 
        else:
            self.estimators = estimators

    def initialize(self):
        self.memory = dict([(i, [0,0]) for i in range(self.bandit.n)])
        self.estimators = [0.5] * self.bandit.n 

    def update_estimator(self, i, reward):
        self.estimators[i] = self.memory[i][0] / (self.memory[i][0] + self.memory[i][1]) 

    def choose_action(self):
        seed = np.random.random()
        if seed < self.eps:
            action = np.random.randint(self.bandit.n)
        else:
            action = np.argmax(self.estimators)
        return action 
    
    def step(self):
        action = self.choose_action()
        reward = self.bandit.generate_reward(action)
        self.update_memory(action, reward)
        self.update_estimator(action, reward)
        return reward


class ThompsonSampling(Policy):
    def __init__(self, memory=None, bandit=None):
        self.bandit = bandit
        if memory is None:
            self.memory = dict([(i,[1,1]) for i in range(self.bandit.n)])
        else:
            self.memory = memory
    
    def initialize(self):
        self.memory = dict([(i,[1,1]) for i in range(self.bandit.n)])

    def choose_action(self):
        rewards = [np.random.beta(self.memory[i][0], self.memory[i][1]) for i in range(self.bandit.n)]
        action = np.argmax(rewards)
        return action 
    
    def step(self):
        action = self.choose_action()
        reward = self.bandit.generate_reward(action)
        self.update_memory(action, reward)
        return reward
    

class UCB(Policy):
    def __init__(self, memory=None, bandit=None):
        self.bandit = bandit
        if memory is None:
            self.memory = dict([(i, [0,0]) for i in range(self.bandit.n)])
        else:
            self.memory = memory
    
    def initialize(self):
        self.memory = dict([(i, [0,0]) for i in range(self.bandit.n)])
        self.counter = [1] * self.bandit.n
        self.estimators = [0.5] * self.bandit.n

    def update_counter(self, i):
        self.counter[i] += 1

    def update_estimator(self, i):
        self.estimators[i] = self.memory[i][0] / np.sum(self.memory[i])

    def choose_action(self, num_step):
        action = np.argmax(self.estimators + np.sqrt(2*np.log(num_step+1)/self.counter))
        return action 
    
    def step(self, num_step):
        action = self.choose_action(num_step)
        reward = self.bandit.generate_reward(action)
        self.update_memory(action, reward)
        self.update_counter(action)
        self.update_estimator(action)
        return reward
    
    def run(self, num_steps):
        self.initialize()
        cum_reward = []
        for i in range(num_steps):
            cum_reward.append(self.step(i))
        return cum_reward
    

    

        


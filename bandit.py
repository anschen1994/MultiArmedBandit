import random


class Bandit():
    def generate_reward(self, i):
        raise NotImplementedError


class BetaBernBandit(Bandit):
    def __init__(self, probs, n):
        assert len(probs) == n 
        self.n = n 
        self.probs = probs

    def generate_reward(self, i):
        seed = random.random()
        if seed < self.probs[i]:
            return 1 
        else:
            return 0
        


        
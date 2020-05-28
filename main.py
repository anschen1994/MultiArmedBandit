from policy import GreedyPolicy, ThompsonSampling, UCB
from bandit import BetaBernBandit
import argparse
import matplotlib.pyplot as plt 
import numpy as np 


parser = argparse.ArgumentParser('MultiArmedBandit')
parser.add_argument('--num-steps', type=int, default=1000)
parser.add_argument('--num-exp', type=int, default=10000)

def simulate_multi(policies, args):
    sample_rewards = []
    for p in policies:
        sample_reward = []
        for i in range(args.num_exp):
            if i % 20 == 0:
                print(i)
            sample_reward.append(p.run(args.num_steps))
        sample_rewards.append(sample_reward)
    plt.plot(np.mean(sample_rewards[0], axis=0), label='Greedy')
    plt.plot(np.mean(sample_rewards[1], axis=0), label='Thompson')
    plt.plot(np.mean(sample_rewards[2], axis=0), label='UCB')
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Mean Reward')
    plt.show()




if __name__ == '__main__':
    probs = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    args = parser.parse_args()
    bandit = BetaBernBandit(probs, len(probs))
    greedy = GreedyPolicy(eps=0.05, bandit=bandit)
    thompson = ThompsonSampling(bandit=bandit)
    ucb = UCB(bandit=bandit)
    simulate_multi([greedy, thompson, ucb], args)
    
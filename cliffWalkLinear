import torch
import gym
import numpy as np
from collections import namedtuple

def basisVector(i,size):
    a = np.zeros(size)
    a[i] = 1.0
    return a

class PolicyNet(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size=64):
        super(PolicyNet, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, hidden_layer_size)
        self.fc2 = torch.nn.Linear(hidden_layer_size, output_size)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = torch.from_numpy(basisVector(x,self.input_size)).float() # this is for cliff-walking ONLY
        return self.softmax(self.fc2(self.fc1(x)))
        # return self.softmax(self.fc2(torch.nn.functional.relu(self.fc1(x))))

    def get_action_and_logp(self, x):
        # print(x)
        action_prob = self.forward(x)
        m = torch.distributions.Categorical(action_prob)
        action = m.sample()
        logp = m.log_prob(action)
        return action.item(), logp

    def act(self, x):
        action, _ = self.get_action_and_logp(x)
        return action

def vpg(env, num_iter=200, num_traj=10, gamma=0.98,
            policy_learning_rate=0.01, policy_saved_path='noCriticCliffWalking.pt'):

    input_size = 4*12 # This is for cliff-walking ONLY
    output_size = env.action_space.n
    Trajectory = namedtuple('Trajectory', 'states actions rewards dones logp')

    def collect_trajectory():
        state_list = []
        action_list = []
        reward_list = []
        dones_list = []
        logp_list = []
        state = env.reset()[0]
        done = False
        truncated = False
        steps = 0

        while (not (done or truncated)) and steps <= 200:
            action, logp = policy.get_action_and_logp(state)
            newstate, reward, done, truncated, info = env.step(action)
            #reward = reward + float(state[0])
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            dones_list.append(done)
            logp_list.append(logp)
            state = newstate
            steps += 1
            # print(steps)

        traj = Trajectory(states=state_list, actions=action_list, rewards=reward_list, logp=logp_list, dones=dones_list)
        return traj

    def calc_returns(rewards, weight):
        T = len(rewards)
        returns = [0]*T
        returns[T-1] = rewards[T-1]
        for i in range(2,T+1):
            returns[T - i] = returns[T-i + 1]*weight + rewards[T-i]
        
        return returns

    policy = PolicyNet(input_size, output_size)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_learning_rate)
    
    mean_return_list = []
    for it in range(num_iter):
        traj_list = [collect_trajectory() for _ in range(num_traj)]
        returns = [calc_returns(traj.rewards, gamma) for traj in traj_list]
        episodic_returns = [calc_returns(traj.rewards, 1) for traj in traj_list]

        policy_loss_terms = [-1. *(gamma**j)* traj.logp[j] * torch.tensor([[returns[i][j]]]) for i, traj in enumerate(traj_list) for j in range(len(traj.actions))]
        
        policy_loss = 1. / num_traj * torch.cat(policy_loss_terms).sum()
        
        # policy_optimizer.zero_grad()
        policy.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()


        mean_return = 1. / num_traj * sum([traj_returns[0] for traj_returns in episodic_returns])
        mean_return_list.append(mean_return)

        if mean_return >= -20:
            print("THIS IS IT:", mean_return)
            torch.save(policy.state_dict(), policy_saved_path)
            quit()

        if it % 10 == 0:
            print('Iteration {}: Mean Return = {}'.format(it, mean_return))
            # print(policy.forward(traj_list[0].states[0]))
            torch.save(policy.state_dict(), policy_saved_path)

    return policy, mean_return_list 









import matplotlib.pyplot as plt
env = gym.make("CliffWalking-v0")
# agent, mean_return_list = vpg(env, num_iter=150, gamma=0.99,num_traj=5) cliffWalking
agent, mean_return_list = vpg(env, num_iter=2000,  gamma=1, num_traj=1)
plt.plot(mean_return_list, color = "crimson")
plt.xlabel('Iteration')
plt.ylabel('Mean Return')
plt.title("No Critic Policy Gradient on cliffWalking")
plt.savefig('noCriticcliffWalking_returns.png', format='png', dpi=300)
state = env.reset()


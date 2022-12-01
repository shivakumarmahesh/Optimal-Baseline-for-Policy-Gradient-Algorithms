import torch
import gym
import numpy as np
from collections import namedtuple

def getGrad(w1,b1,w2,b2):
    return (torch.cat((torch.flatten(w1),torch.flatten(b1),torch.flatten(w2),torch.flatten(b2))))



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

def gradInnerProductOverTime(env, num_traj=10, gamma=0.98,
            policy_learning_rate=0.01, policy_saved_path='xyz.pt'):

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

        while (not (done or truncated)) and steps <= 100:
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
    policy.load_state_dict(torch.load("optimalCliffWalkController.pt"))
    # policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_learning_rate)
    
    mean_return_list = []
    
    traj_list = [collect_trajectory() for _ in range(num_traj)]
    # gradList = []
    innerProdGradList = []
    # returns = [calc_returns(traj.rewards, gamma) for traj in traj_list]
    # episodic_returns = [calc_returns(traj.rewards, 1) for traj in traj_list]
    # policy_loss_terms = [-1. *(gamma**j)* traj.logp[j] * torch.tensor([[returns[i][j]]]) for i, traj in enumerate(traj_list) for j in range(len(traj.actions))]
    # policy_loss = 1. / num_traj * torch.cat(policy_loss_terms).sum()

    policy_loss = traj_list[0].logp[0]
        
    # policy_optimizer.zero_grad()
    policy.zero_grad()
    policy_loss.backward()
    gradAtTimeZero = getGrad(policy.fc1.weight.grad, policy.fc1.bias.grad, policy.fc2.weight.grad, policy.fc2.bias.grad)
    innerProdGradList.append(torch.dot(gradAtTimeZero,gradAtTimeZero).item())
    print(len(traj_list[0].actions))

    for t in range(1,len(traj_list[0].actions)):

        policy_loss = traj_list[0].logp[t]
        # print("Traj lenght:", len(traj_list[0].actions))
        
        # policy_optimizer.zero_grad()
        policy.zero_grad()
        policy_loss.backward()

        currentGrad = getGrad(policy.fc1.weight.grad, policy.fc1.bias.grad, policy.fc2.weight.grad, policy.fc2.bias.grad)
        innerProdGradList.append(torch.dot(gradAtTimeZero,currentGrad).item())


    # mean_return = 1. / num_traj * sum([traj_returns[0] for traj_returns in episodic_returns])
    # mean_return_list.append(mean_return)


    # print(policy_loss.parameters())
    # print(policy.forward(traj_list[0].states[0]))
    # torch.save(policy.state_dict(), policy_saved_path)

    return policy, innerProdGradList






import matplotlib.pyplot as plt
env = gym.make("CliffWalking-v0")

agent, innerProdGradList = gradInnerProductOverTime(env, gamma=1, num_traj=1)
plt.plot(innerProdGradList, color = "crimson")
plt.xlabel('t')
plt.ylabel('Inner Product between GradLogs over Time')
plt.title("⟨∇logπ(a0|s0),∇logπ(a_t|s_t)⟩  under an Optimal Policy")
plt.savefig('OptimalPolicyGradLogsOverTime.png', format='png', dpi=300)
state = env.reset()


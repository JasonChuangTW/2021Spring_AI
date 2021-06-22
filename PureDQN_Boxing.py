from numpy import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random as rand

#Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 10   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('Boxing-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
Input_shape = env.observation_space.shape
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
device = 'cuda'

class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(Input_shape[1])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(Input_shape[0])))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, N_ACTIONS)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.permute(0,3,1,2)
        #x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class Memory_Buffer(object):
    def __init__(self, memory_size=1000):
        self.buffer = []
        self.memory_size = MEMORY_CAPACITY
        self.next_idx = 0
        
    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= self.memory_size: # buffer not full
            self.buffer.append(data)
            self.next_idx +=1
        else: # buffer is full
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.size() - 1)
            data = self.buffer[idx]
            state, action, reward, next_state, done= data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return torch.FloatTensor(states),torch.LongTensor(actions),torch.FloatTensor(rewards),torch.FloatTensor(next_states)
    
    def size(self):
        return len(self.buffer)

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory = Memory_Buffer()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        self.memory.push(s,a,r,s_,0)

    def learn(self):
        print("-----------[learning]--------------")
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            print("-----------[update target]----------------")
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        b_s,b_a,b_r,b_s_= self.memory.sample(BATCH_SIZE)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1,np.reshape(b_a,(-1,32)))  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(10000):
    s = env.reset()
    ep_r = 0
    while True:
        if i_episode%5 == 0:
            env.render()
        a = dqn.choose_action(s)
        # take action
        s_, r, done, info = env.step(a)

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory.next_idx == MEMORY_CAPACITY-2:
            dqn.learn()

        if done:
            break
        s = s_
# imports
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from models import ModelCnn
from replay_buffer import ReplayBuffer4D
import time

class Agent(object):
    def __init__(self, action_space):

        # TRAINING HYPERPARAMS
        self.learning_rate = 0.0025
        self.gamma = 0.9
        self.update_freq = 1000
        self.batch_size = 128
        self.epsilon = 1.0
        self.epsilon_decay = 0.999995 
        self.epsilon_min = 0.1

        # create replay buffer
        self.buffer_capacity = 100000
        self.input_height = 251
        self.input_width = 251
        self.input_depth = 2

        # save the model
        self.model_name = 'agent_g2' # f3 - buffer # 4- 0.9 # 2 - 0.98 0.0025 # 0.00025 # f6 lost mean
        self.model_path = 'checkpoints/' + self.model_name + '.pth' 
        print('---------------------')
        print('Model: ', self.model_name )
        print('---------------------')
        # LOAD MODEL
        self.action_space = action_space

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device name: ', torch.cuda.get_device_name(self.device))

        self.local_network = ModelCnn()
        self.target_network = ModelCnn()

        self.local_network.to(self.device)
        self.target_network.to(self.device)
        self.local_network.train()
        self.target_network.eval()

        self.optimizer = optim.RMSprop(self.local_network.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.SGD(self.local_network.parameters(), lr=self.learning_rate, momentum=0.9)
        #self.optimizer = optim.Adam(self.local_network.parameters(), lr=self.learning_rate)

        self.criterion = nn.SmoothL1Loss() #nn.MSELoss()

        self.replay = ReplayBuffer4D(self.buffer_capacity, self.input_height, self.input_width,
                                    self.batch_size, self.input_depth)

        self.start_learn = False
        self.training_loss = 0.0

        self.reward_hist = []
        self.best_mean_reward = 0.0

    def act(self, state):

        # implement epsilon greedy
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

        #return action
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()

        state = np.expand_dims(state, axis=0)
        #state = np.expand_dims(state, axis=0)
        state = torch.from_numpy(state)
        state = state.float()
        state = state.to(self.device)

        # print('--')
        self.local_network.eval()
        action_values = self.local_network(state)
        self.local_network.train()
        # print('------')
        action = np.argmax(action_values.cpu().detach().numpy())

        #print('DQ RL action')
        #print('self.local_network(state): ', action_values)
        #print('local_network out ', action_values, '  ----->    ', np.argmax(action_values.cpu().detach().numpy()))

        return action

    def step(self, state, action, reward, done, next_state):

        # update the target network
        if self.replay.position % self.update_freq == 0:
            #print('replaced')
            self.target_network.load_state_dict(self.local_network.state_dict())

            # save the network
            #torch.save(self.target_network.state_dict(), 'checkpoints/local_cartpole-V1.pth')

        # keep to the memory
        self.replay.push(state, action, reward, done, next_state)

        # save the model with best mean reward
        self.reward_hist.append(reward)
        temp_mean = np.mean(np.array(self.reward_hist[-20:]))
        if  self.best_mean_reward <= temp_mean:
            self.best_mean_reward = temp_mean
            torch.save(self.target_network.state_dict(), self.model_path)
        
        if self.replay.position > self.batch_size:
            self.start_learn = Trues

        # train the local_network
        if self.start_learn:
            # learn if enough sample is available
            states_batch, actions_batch, rewards_batch, dones_batch, next_states_batch = self.replay.sample(self.batch_size)

            # ???
            #states_batch = np.expand_dims(states_batch, axis=1)
            #next_states_batch = np.expand_dims(next_states_batch, axis=1)

            states_batch = torch.from_numpy(states_batch)
            next_states_batch = torch.from_numpy(next_states_batch)
            actions_batch = torch.from_numpy(actions_batch)
            rewards_batch = torch.from_numpy(rewards_batch)
            dones_batch = torch.from_numpy(dones_batch)

            states_batch = states_batch.float().to(self.device)
            next_states_batch = next_states_batch.float().to(self.device)
            rewards_batch = rewards_batch.float().to(self.device)
            actions_batch = actions_batch.to(self.device)
            dones_batch = dones_batch.float().to(self.device)

            # ???

            if np.random.rand() < 0.5:
                next_actions = self.local_network(next_states_batch).max(1)[1]
                next_qval = self.target_network(next_states_batch).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            else:
                next_qval = self.target_network(next_states_batch).detach().max(1)[0]

            target_qval = rewards_batch + self.gamma*next_qval*(1-dones_batch)

            output_qval = self.local_network(states_batch).gather(1, actions_batch.unsqueeze(-1)).squeeze(-1)

            loss = self.criterion(output_qval, target_qval)

            loss = loss.mean()
            self.training_loss = loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

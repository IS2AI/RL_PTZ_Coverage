# imports
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models import DuellingModelCnn
from replay_buffer import PrioReplayBuffer4D

class Agent(object):
    def __init__(self, action_space):

        # TRAINING HYPERPARAMS
        self.learning_rate = 0.0002
        self.gamma = 0.95
        self.update_freq = 1600
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_decay = 0.999998
        self.epsilon_min = 0.1

        # create replay buffer
        self.buffer_capacity = 100000
        self.input_height = 251
        self.input_width = 251
        self.input_depth = 2

        # save the model8
        self.model_name = 'ptz_FF26'
        self.model_path = 'checkpoints/' + self.model_name + '.pth' 

        print('--------------------')
        print('Model: ', self.model_name )
        print('--------------------')

        # LOAD MODEL
        self.action_space = action_space

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print('Device name: ', torch.cuda.get_device_name(self.device))

        self.local_network = DuellingModelCnn()
        self.target_network = DuellingModelCnn()

        self.local_network.to(self.device)
        self.target_network.to(self.device)
        self.local_network.train()
        self.target_network.eval()

        self.optimizer = optim.RMSprop(self.local_network.parameters(), lr=self.learning_rate)
        #self.optimizer = optim.SGD(self.local_network.parameters(), lr=self.learning_rate, momentum=0.9)
        #self.optimizer = optim.Adam(self.local_network.parameters(), lr = self.learning_rate)

        self.criterion = nn.SmoothL1Loss(reduction='none') # nn.MSELoss()
        self.replay = PrioReplayBuffer4D(self.buffer_capacity, self.input_height, self.input_width,
                                            self.batch_size, self.input_depth)
                                    
        self.start_learn = False
        self.training_loss = 0.0
        
        self.reward_hist = []
        self.best_mean_reward = -float('inf')

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
            torch.save(self.local_network.state_dict(), self.model_path)
            
            sample_input = torch.from_numpy(np.ones((1,self.input_depth,self.input_height,self.input_width))).float().to(self.device)
            traced_script_module = torch.jit.trace(self.local_network, sample_input)
            traced_script_module.save('checkpoints/' + self.model_name + '.pt')            


        if self.replay.position_filled > self.batch_size*100:
            self.start_learn = True

        # train the local_network
        if self.start_learn:
            # learn if enough sample is available
            states_batch, actions_batch, rewards_batch, dones_batch, next_states_batch, importance, indices = self.replay.sample(self.batch_size)

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
            
            # square
            td_errors = (target_qval - output_qval)**2
            
            # huber loss
            td_errors = self.criterion(target_qval, output_qval)
            #print(td_errors.shape)
            #if 

            weighed_td_errors = (torch.FloatTensor(importance**(self.replay.beta)).to(self.device))*td_errors

            loss = weighed_td_errors.mean()

            self.training_loss = loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update priorities
            self.replay.update_priorities(indices, abs(td_errors) + 0.01)


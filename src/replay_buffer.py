"""
Created on Wed Feb 12 2020
@author: Daulet Baimukashev
"""
import random
import numpy as np

class ReplayBuffer1D(object):
    def __init__(self, capacity, im_height, batch_size):

        self.capacity = capacity
        self.position_filled = 0
        self.position = 0
        
        self.im_height = im_height
        self.batch_size = batch_size
        
        # one experience
        self.states = np.zeros((self.capacity, self.im_height), dtype = np.float32)
        self.actions = np.zeros(self.capacity, dtype = np.int64)
        self.rewards = np.zeros(self.capacity, dtype = np.float32)
        self.dones = np.zeros(self.capacity, dtype = np.int64)
        self.next_states = np.zeros((self.capacity, self.im_height), dtype = np.float32)

        # many experiences for one batch
        self.states_batch = np.zeros((self.batch_size, self.im_height ), dtype = np.float32)
        self.actions_batch = np.zeros(self.batch_size, dtype = np.int64)
        self.rewards_batch = np.zeros(self.batch_size, dtype = np.float32)
        self.dones_batch = np.zeros(self.batch_size, dtype = np.int64)
        self.next_states_batch = np.zeros((self.batch_size, self.im_height), dtype = np.float32)


    def push(self, state, action, reward, done, next_state):

            # find the position for the incoming sample
            self.position = (self.position + 1)%self.capacity

            self.states[self.position] = state
            self.actions[self.position] = action
            self.rewards[self.position] = reward
            self.dones[self.position] = done
            self.next_states[self.position] = next_state
            
            # track filled buffer size
            self.position_filled = min(self.position_filled + 1, self.capacity)

    def sample(self, batch_size):

        for i in range(batch_size):
            #
            index = random.randint(0, self.position_filled - 1)

            self.states_batch[i] = self.states[index]
            self.actions_batch[i] = self.actions[index]
            self.rewards_batch[i] = self.rewards[index]
            self.dones_batch[i] = self.dones[index]
            self.next_states_batch[i] = self.next_states[index]

            # transpose ???
            # new axis
            #self.states_batch = np.transpose(self.states_batch, axes=(0, 2, 3, 1))
            #self.next_states_batch = np.transpose(self.next_states_batch, axes=(0, 2, 3, 1))

        return self.states_batch, self.actions_batch, self.rewards_batch, self.dones_batch, self.next_states_batch

    def __len__(self):
        return self.position_filled



'''
4D Buffer for image
'''
class ReplayBuffer4D(object):
    def __init__(self, capacity, im_height, im_width, batch_size, im_channels):

        self.capacity = capacity
        self.position_filled = 0
        self.position = 0

        self.im_height = im_height
        self.batch_size = batch_size
        
        # ADD MORE DIMENSIONS
        self.im_channels = im_channels
        self.im_width = im_width
        
        # one experience
        self.states = np.zeros((self.capacity, self.im_channels, self.im_height, self.im_width), dtype = np.float32)
        self.actions = np.zeros(self.capacity, dtype = np.int64)
        self.rewards = np.zeros(self.capacity, dtype = np.float32)
        self.dones = np.zeros(self.capacity, dtype = np.int64)
        self.next_states = np.zeros((self.capacity, self.im_channels, self.im_height, self.im_width), dtype = np.float32)

        # many experiences for one batch
        self.states_batch = np.zeros((self.batch_size, self.im_channels, self.im_height, self.im_width), dtype = np.float32)
        self.actions_batch = np.zeros(self.batch_size, dtype = np.int64)
        self.rewards_batch = np.zeros(self.batch_size, dtype = np.float32)
        self.dones_batch = np.zeros(self.batch_size, dtype = np.int64)
        self.next_states_batch = np.zeros((self.batch_size, self.im_channels, self.im_height, self.im_width), dtype = np.float32)

    def push(self, state, action, reward, done, next_state):

            # find the position for the incoming sample
            self.position = (self.position + 1)%self.capacity

            self.states[self.position] = state
            self.actions[self.position] = action
            self.rewards[self.position] = reward
            self.dones[self.position] = done
            self.next_states[self.position] = next_state

            # track filled buffer size
            self.position_filled = min(self.position_filled + 1, self.capacity)

    def sample(self, batch_size):

        for i in range(batch_size):
            #
            index = random.randint(0, self.position_filled - 1)

            self.states_batch[i] = self.states[index]
            self.actions_batch[i] = self.actions[index]
            self.rewards_batch[i] = self.rewards[index]
            self.dones_batch[i] = self.dones[index]
            self.next_states_batch[i] = self.states[index]

        return self.states_batch, self.actions_batch, self.rewards_batch, self.dones_batch, self.next_states_batch

    def __len__(self):
        return self.position_filled


###################################################################
# Prioritize Replay - 4D
###################################################################

class PrioReplayBuffer4D(object):
    def __init__(self, capacity, im_height, im_width, batch_size, im_channels):

        self.prio_alpha = 0.6
        self.beta_start = 0.4
        self.beta = 0.4
        self.beta_count = 0
        self.beta_decay_steps = 400000    

        self.capacity = capacity
        self.position_filled = 0
        self.position = -1
        
        self.im_height = im_height
        self.batch_size = batch_size
        self.priorities = np.zeros(self.capacity, dtype = np.float32)

        # ADD MORE DIMENSIONS
        self.im_channels = im_channels
        self.im_width = im_width

        # one experience
        self.states = np.zeros((self.capacity, self.im_channels, self.im_height, self.im_width), dtype = np.float32)
        self.actions = np.zeros(self.capacity, dtype = np.int64)
        self.rewards = np.zeros(self.capacity, dtype = np.float32)
        self.dones = np.zeros(self.capacity, dtype = np.int64)
        self.next_states = np.zeros((self.capacity, self.im_channels, self.im_height, self.im_width), dtype = np.float32)

        # many experiences for one batch
        self.states_batch = np.zeros((self.batch_size, self.im_channels, self.im_height, self.im_width), dtype = np.float32)
        self.actions_batch = np.zeros(self.batch_size, dtype = np.int64)
        self.rewards_batch = np.zeros(self.batch_size, dtype = np.float32)
        self.dones_batch = np.zeros(self.batch_size, dtype = np.int64)
        self.next_states_batch = np.zeros((self.batch_size, self.im_channels, self.im_height, self.im_width), dtype = np.float32)

    def push(self, state, action, reward, done, next_state):

            # find the position for the incoming sample
            self.position = (self.position + 1)%self.capacity

            self.states[self.position] = state
            self.actions[self.position] = action
            self.rewards[self.position] = reward
            self.dones[self.position] = done
            self.next_states[self.position] = next_state
            
            # hands on AI game 233
            max_prio = self.priorities.max() if self.position_filled==0 else 1.0
            self.priorities[self.position] = max_prio

            # track filled buffer size
            self.position_filled = min(self.position_filled + 1, self.capacity)

    def get_probs(self, prio_scale):
        scaled_prios = np.array(self.priorities) ** prio_scale
        sample_prios = scaled_prios/ sum(scaled_prios)
        return sample_prios

    def get_importance(self, probs):
        importance = 1/self.position_filled * 1/probs
        importance_normalized = importance/max(importance)
        return importance_normalized

    def update_priorities(self, indices, priorities):
        for ind, prio in zip(indices, priorities):
            self.priorities[ind] = prio

    def sample(self, batch_size):
        
        self.beta_count += 1
        self.beta = min(1.0, self.beta_start + self.beta_count * (1.0 - self.beta_start) / self.beta_decay_steps)

        sample_probs = self.get_probs(self.prio_alpha)
        sample_indices = random.choices(range(self.position_filled),k=batch_size, weights = sample_probs[0:self.position_filled])
        importance = self.get_importance(sample_probs[sample_indices])

        for i in range(batch_size):           
            index = sample_indices[i]
            self.states_batch[i] = self.states[index]
            self.actions_batch[i] = self.actions[index]
            self.rewards_batch[i] = self.rewards[index]
            self.dones_batch[i] = self.dones[index]
            self.next_states_batch[i] = self.next_states[index]

        return self.states_batch, self.actions_batch, self.rewards_batch, self.dones_batch, self.next_states_batch, importance, sample_indices


    def __len__(self):
        return self.position_filled




'''
Reinforce
'''

class PolicyReplayBuffer(object):
    def __init__(self, capacity, im_height, im_width, im_depth, batch_size):

        self.capacity = capacity
        self.position_filled = 0
        self.position = 0

        self.im_height = im_height
        self.batch_size = batch_size

        # one experience
        self.states = np.zeros((self.capacity, self.im_height, im_width, im_depth), dtype = np.float32)
        self.actions = np.zeros(self.capacity, dtype = np.int64)
        self.rewards = np.zeros(self.capacity, dtype = np.float32)
        #self.dones = np.zeros(self.capacity, dtype = np.int64)
        #self.next_states = np.zeros((self.capacity, self.im_height), dtype = np.float32)

        # many experiences for one batch
        self.states_batch = np.zeros((self.batch_size, self.im_height, im_width, im_depth), dtype = np.float32)
        self.actions_batch = np.zeros(self.batch_size, dtype = np.int64)
        self.rewards_batch = np.zeros(self.batch_size, dtype = np.float32)
        #self.dones_batch = np.zeros(self.batch_size, dtype = np.int64)
        #self.next_states_batch = np.zeros((self.batch_size, self.im_height), dtype = np.float32)


    def push(self, state, action, reward):

        # find the position for the incoming sample
        self.position = (self.position + 1)%self.capacity

        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        #self.dones[self.position] = done
        #self.next_states[self.position] = next_state

        # track filled buffer size
        self.position_filled = min(self.position_filled + 1, self.cap)

    def sample(self):

        '''
        for i in range(self.position_filled):     # position_filled instead of batch_size
            
            index = i #random.randint(0, self.position_filled - 1)

            self.states_batch[i] = self.states[index]
            self.actions_batch[i] = self.actions[index]
            self.rewards_batch[i] = self.rewards[index]
            #self.dones_batch[i] = self.dones[index]
            #self.next_states_batch[i] = self.next_states[index]

            # transpose ???
            # new axis
            #self.states_batch = np.transpose(self.states_batch, axes=(0, 2, 3, 1))
            #self.next_states_batch = np.transpose(self.next_states_batch, axes=(0, 2, 3, 1))
        '''

        #print('hi - ', self.position_filled)
        #:print(self.states_batch.shape)
        self.states_batch = self.states[0:self.position_filled]
        self.actions_batch = self.actions[0:self.position_filled]
        self.rewards_batch = self.rewards[0:self.position_filled]
        #print(self.states_batch.shape)

        return self.states_batch, self.actions_batch, self.rewards_batch #, self.dones_batch, self.next_states_batch

    def __len__(self):
        return self.position_filled

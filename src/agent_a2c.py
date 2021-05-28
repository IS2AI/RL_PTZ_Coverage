import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models import ActorCriticCnn
from replay_buffer import PolicyReplayBuffer
from lib import wrappers
import gym 
import gym_viewshed

class Agent(object):
    def __init__(self, action_space):

        #TRAINING HYPERPARAMS
        self.learning_rate = 0.0001
        self.gamma = 0.95
        self.batch_size = 500

        # create replay buffer
        self.capacity = 500
        self.buffer_capacity = 500
        self.input_height = 251
        self.input_width = 251
        self.input_depth = 2

        #save models
        self.model_name = 'ping_HH3'
        self.model_path = 'checkpoints/' + self.model_name + '.pth'

        print('---------')
        print('Model: ', self.model_name)
        print('---------')

        # Load model
        self.action_space = action_space

        self.device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print("Device name: ", torch.cuda.get_device_name(self.device)) 

        self.local_network = ActorCriticCnn()
        self.local_network.train()
        self.local_network.to(self.device)

        self.optimizer = optim.Adam(self.local_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.replay = PolicyReplayBuffer(self.capacity, self.input_height, self.input_width, self.input_depth , self.batch_size)

        self.start_learn = False
        self.training_loss = 0.0

        self.reward_hist = []
        self.best_mean_reward = -float('inf')

        self.episode = 0
        self.replaced = 0

        self.epochs = 200
        # not used
        self.epsilon = 1.0

    def worker(self, t, worker_model, counter, params):
    
        env_count = 0
        
        worker_env = gym.make('gym_viewshed:viewshed-v5')
        worker_env.seed(t)
       
        worker_env.reset()

        worker_opt = optim.RMSprop(lr=2e-4, params=worker_model.parameters())
        worker_opt.zero_grad()

        for i in range(self.epochs):
            self.episode = i
            #print('1 worker started - ', i, t)
            worker_opt.zero_grad()
            
            values, logprobs, rewards, G = self.run_episode(worker_env, worker_model)
            #print('2 run epidode - ', i, t)
            actor_loss, critic_loss, eplen = self.update_params(worker_opt, values, logprobs, rewards, G)
            #print('3 update - ', i, t)
            counter.value = counter.value + 1
            print('Epoch {}, Worker {}, Reward {}'.format(i, t, np.mean(np.array(rewards))))


    def run_episode(self, worker_env, worker_model, N_steps=50):

        values, logprobs, rewards = [], [], []
        done = False
        j = 0
        G = torch.Tensor([0])
        #state = torch.from_numpy(worker_env.state).float().to(self.device)

        state = torch.from_numpy(worker_env.reset()).float().to(self.device)

        while (done == False):
            j += 1

            action_probs, value = worker_model(state)
            values.append(value)
            #a = action_probs.cpu().data.numpy()
            #print(action_probs, value)
            action = np.random.choice(np.array([0,1,2,3,4,5]), p=action_probs.cpu().data.numpy())

            logprobs.append(action_probs[action])

            next_state, reward_env, done, info = worker_env.step(action)

            state = torch.from_numpy(next_state).float().to(self.device)

            if done:
                reward = reward_env
                worker_env.reset()
            else:
                reward = reward_env
                G = value.detach()

            rewards.append(reward)

        return values, logprobs, rewards, G


    def update_params(self, worker_opt, values, logprobs, rewards, G, clc=0.1, gamma=0.95):

        rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
        logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
        values = torch.stack(values).flip(dims=(0,)).view(-1)

        Returns  = []
        ret_ = G
        for r in range(rewards.shape[0]):
            ret_ = rewards[r] + gamma * ret_
            Returns.append(ret_)
        Returns = torch.stack(Returns).view(-1)
        Returns = F.normalize(Returns, dim=0)
        #Returns_2 = self.discount_rewards(rewards)

        actor_loss = -1*logprobs * (Returns - values.detach())
        critic_loss = torch.pow(values - Returns, 2)

        loss = actor_loss.sum() + clc*critic_loss.sum()
        loss.backward()
        worker_opt.step()

        return actor_loss, critic_loss, len(rewards)

    def act(self, state):

        state = torch.from_numpy(state)
        state = state.float()
        state = state.to(self.device)

        self.local_network.eval()
        action_probs, values = self.local_network(state)
        self.local_network.train()

        action = np.random.choice(np.array([0,1,2,3,4,5]), p=action_probs.cpu().detach().numpy()) #p=action_probs.cpu().detach().numpy())

        return action


    def Xstep(self, state, action, reward):

        self.replay.push(state, action, reward)

        # save the model with best mean reward
        self.reward_hist.append(reward)
        temp_mean = np.mean(np.array(self.reward_hist[-20:]))
        if  self.best_mean_reward <= temp_mean:
            self.best_mean_reward = temp_mean
            #torch.save(self.target_network.state_dict(), self.model_path)
            torch.save(self.local_network.state_dict(), self.model_path)


    def Xtraining(self):

        # training
        states_batch, actions_batch, rewards_batch = self.replay.sample()

        # reset buffer
        self.replay.position = 0
        self.replay.position_filled = 0

        # calculate discounted rewards
        # rewards_batch = rewards_batch.flip(dims=(0,))
        states_batch = torch.from_numpy(states_batch)
        actions_batch = torch.from_numpy(actions_batch)
        rewards_batch = torch.from_numpy(rewards_batch)

        rewards_batch = rewards_batch.flip(dims=(0,))
        rewards_batch = self.discount_rewards(rewards_batch)

        states_batch = states_batch.float().to(self.device)
        rewards_batch = rewards_batch.float().to(self.device)
        actions_batch = actions_batch.to(self.device)

        # loss ???
        self.local_network.eval()
        pred_batch = self.local_network(states_batch)
        self.local_network.train()
        prob_batch = pred_batch.gather(dim=1, index=actions_batch.long().view(-1,1)).squeeze()

        loss = -torch.sum(rewards_batch * torch.log(prob_batch))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def discount_rewards(self, rewards):

        lenr = len(rewards)
        disc_return = torch.pow(self.gamma, torch.arange(lenr).float()) * rewards
        disc_return /= disc_return.max()

        return disc_return


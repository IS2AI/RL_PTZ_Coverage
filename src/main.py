import numpy as np
import time
import gym
import torch
from tensorboardX import SummaryWriter
import sys
import copy
import matplotlib.pyplot as plt


def main():

    # env
    env_count = 0
    for env in gym.envs.registry.env_specs:
        if 'viewshed' in env:
            print('Remove {} from registry'.format(env))
            del gym.envs.registry.env_specs[env]

    import gym_viewshed

    env = gym.make('gym_viewshed:viewshed-v5')
    env.seed(10)

    #render = lambda : plt.imshow(env.render(mode='rgb_array'))

    #from agent_random import Agent
    #agent = Agent(env.action_space)

    # DQ agent
    #from agent_dq import Agent
    #agent = Agent(env.action_space)

    from agent_double_dqn_per import Agent
    agent = Agent(env.action_space)

    #from agent_doubledqn import Agent
    #agent = Agent(env.action_space)

    # select train-test mode
    mode = int(sys.argv[1])

    if mode == 0: # train mode
        print('Mode - Training ...')
        max_episode = 5000
        max_iteration = 1500
        episode_score_list = []

        start_time = time.time()
        writer = SummaryWriter(comment='__' + agent.model_name)

        for i in range (max_episode+1):
            state = env.reset()
            episode_score = 0

            for j in range(max_iteration):
                
                #env.render()         
                # -- prev --
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, done, next_state)
                state = next_state
                episode_score += reward
                
                if done:
                    break

            # printing the info/tensorboard
            elapsed = (time.time() - start_time)/60
            episode_score_list.append(episode_score)
            if i%10 == 0:
                print('Episode {}, Iteration {}, Epsilon {}, Beta {}, Score {}, Elapsed minutes {},  less {}'.format(i, j, agent.epsilon, agent.replay.beta, episode_score, elapsed, env.info))
            items_display = {"Mean reward": np.mean(np.array(episode_score_list[-20:])), "Episode reward": episode_score, "loss": agent.training_loss, "eps": agent.epsilon}
            for name, val in items_display.items():
                writer.add_scalar(name, val)

        writer.close()
        print('>>> Training finished.. Max episode_score_list: ',  max(episode_score_list))

    else:
        print('Mode - Testing ...')
        # TEST mode
        max_episode = 3
        max_iteration = 500  
        episode_score_list = []
        start_time = time.time()

        agent.epsilon = 0.1
        agent.min = 0.1
        #local_network.load_state_dict(torch.load('checkpoints/testing_A1.pth')) #local_cartpole-V1
        
        agent.local_network = torch.jit.load('checkpoints/ptz_FF7.pt')
        for i in range (max_episode):

            state = env.reset()
            episode_score = 0

 
            for j in range(max_iteration):
                env.render()
                time.sleep(0.1)
                
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                
                
                episode_score += reward
                if done:
                    break

            # printing the info/tensorboard
            elapsed = (time.time() - start_time)/60
            episode_score_list.append(episode_score)
            print('Episode {}, Iteration {}, Score {}, Elapsed minutes {}'.format(i, j, episode_score, elapsed))

        print('>>> Testing finished. Max episode_score_list: ',  max(episode_score_list))




if __name__ == '__main__':
    main()

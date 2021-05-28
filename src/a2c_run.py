import sys
import multiprocessing as mp
import numpy as np
import time
import gym
import torch
from tensorboardX import SummaryWriter

from lib import wrappers

def main():

    mode = int(sys.argv[1])

      
    # AGENTS
    from agent_a2c import Agent

    # SETUP
    agent = Agent(6)

    if mode == 0: # train mode

        # Start the Multiprocessing
        MasterNode = agent.local_network
        MasterNode.share_memory()
        processes = []
        params = {'epochs':10, 'n_workers':16,}
        counter = mp.Value('i',0)
        for i in range(params['n_workers']):
            p = mp.Process(target=agent.worker, args=(i,MasterNode,counter,params))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        for p in processes:
            p.terminate()

        print('>>> Training finished ..... ')
        torch.save(agent.local_network.state_dict(), agent.model_path)

    else:
        
        # ENV - works

        env = gym.make('gym_viewshed:viewshed-v5')
        env.seed(100)
        print('Mode - Testing ...')
        
        # TEST mode
        max_episode = 10
        max_iteration = 500
        episode_score_list = []
        start_time = time.time()

        agent.epsilon = 0.1
        agent.min = 0.1
        agent.local_network.load_state_dict(torch.load('checkpoints/ping_HH3.pth'))

        for i in range (max_episode):

            state = env.reset()
            episode_score = 0

            for j in range(max_iteration):
                env.render()

                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                #agent.step(state, action, reward, done, next_state)
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


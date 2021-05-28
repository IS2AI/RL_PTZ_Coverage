class Agent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.epsilon = 1.0
        self.training_loss = 1.0

    def act(self, state):
        return self.action_space.sample()
    
    def step(self,state, action, reward, done, next_state):
        pass

    def close(self):
        pass
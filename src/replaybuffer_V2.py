import random
import numpy as np

random.seed(10)

class ReplayBuffer:
  
    def __init__(self, max_size = 200):
        """
            Create Replay buffer which stores a sequence of transitions. 
            A transition is a tuple: (state, action, reward, next state, trunc).
            Is used to compute the 1-step TD-learning update rule.
            Variable trunc is a boolean that assumes a positive value after max steps (max_it).

            Parameters
            ----------
            max_size: int
                Max number of transitions to store in the buffer. When the buffer
                overflows the old memories are dropped.
            """
        self.num_exp=0
        self.buffer = np.zeros((int(max_size), 9))
        self.max_size = int(max_size) # buffer size: how many transitions it can store at most
        self.counter = 0
        
    def count(self):
        return self.counter
    
    def add_transition(self, state, action, reward, next_state, trunc):
        self.buffer[self.counter % self.max_size, :] = np.array([*state, action, reward, *next_state, trunc])
        self.counter += 1


    def sample_transition(self, batch_size):
        """Samples a batch of experiences."""
        if self.counter < batch_size:
            size = self.counter
        else:
            size = batch_size
            
        if self.counter >= self.max_size:
            batch_transitions = self.buffer[np.random.randint(low = 0, high = self.max_size, size = size),:]
        elif self.counter < self.max_size:
            batch_transitions = self.buffer[np.random.randint(low = 0, high = self.counter, size = size),:]

        return batch_transitions[:,0:3], batch_transitions[:,3],batch_transitions[:,4],batch_transitions[:,5:8], batch_transitions[:,8] 

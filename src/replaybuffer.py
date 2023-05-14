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
        self.buffer = []
        # self.state_buffer = []
        # self.action_buffer = []
        # self.reward_buffer = []
        # self.next_state_buffer = []
        self.max_size = max_size # buffer size: how many transitions it can store at most

    def count(self):
        return self.num_exp
    
    def add_transition(self, state, action, reward, next_state, trunc):
        transition = (state, action, reward, next_state, trunc)
        if self.num_exp < self.max_size:
            self.buffer.append(transition)
            # self.state_buffer.append(state)
            # self.action_buffer.append(action)
            # self.reward_buffer.append(reward)
            # self.next_state_buffer.append(next_state)
            self.num_exp +=1
        else:
            self.buffer.pop(0)
            self.buffer.append(transition)
            # self.state_buffer.pop(0)
            # self.action_buffer.pop(0)
            # self.reward_buffer.pop(0)
            # self.next_state_buffer.pop(0) # why popleft doesn't work?

            # self.state_buffer.append(state)
            # self.action_buffer.append(action)
            # self.reward_buffer.append(reward)
            # self.next_state_buffer.append(next_state)
            

    def sample_transition(self, batch_size):
        """
        Sample a batch of experiences.
        """
        #transitions = []
        size = 0
        if self.num_exp < batch_size:
            size = self.num_exp
        else:
            size = batch_size
        
        batch_transitions=random.sample(self.buffer, size)

        state_batch, action_batch, reward_batch, next_state_batch, trunc = map(np.stack, zip(*batch_transitions))
        # state_batch = random.sample(self.state_buffer, size)
        # action_batch = random.sample(self.action_buffer, size)
        # reward_batch = random.sample(self.reward_buffer, size)
        # next_state_batch = random.sample(self.next_state_buffer, size)
        # trunc = [False for _ in range(size)]


        # for i in size:
        #     transitions.append((self.state_buffer[i], self.action_buffer[i], self.reward_buffer[i], self.next_state_buffer[i], False))
        return state_batch, action_batch, reward_batch, next_state_batch, trunc
 

import numpy as np
# Replay buffer class
# There is a task here!

class ReplayBuffer():
    def __init__(self,
                 max_size=10000,
                 batch_size=64):
        """
        Initialize class

        ss_mem = state buffer
        as_mem = action buffer
        rs_mem = reward buffer
        ps_mem = probability buffer
        ds_mem = discount factor buffer

        max_size = maximum buffer size
        batch_size = sample batch size
        _idx = buffer index
        size = ongoing buffer size
        """
        self.ss_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ds_mem = np.empty(shape=(max_size), dtype=np.ndarray)

        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0

    def store(self, sample):
        """
        get values from the samples

        sample = tuple (s, a, r, p, d)
        s = state
        a = action
        r = reward
        p = new_state
        d = flag for terminal state

        ss_mem = memory buffer for current states
        as_mem = memory buffer for actions
        rs_mem = memory buffer for rewards
        ps_mem = memory buffer for new states
        ds_mem = memory buffer for terminal state flag
        """
        s, a, r, p, d = sample

        # TODO: Complete the function store in the relay buffer class
        self.ss_mem[self._idx] =  s# To complete
        self.as_mem[self._idx] =  a# To complete
        self.rs_mem[self._idx] =  r# To complete
        self.ps_mem[self._idx] =  p# To complete
        self.ds_mem[self._idx] =  d# To complete

        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        """
        store samples in the buffer

        idxs = index
        experiences = samples in buffer
        """
        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(
            self.size, batch_size, replace=False)
        experiences = np.vstack(self.ss_mem[idxs]), \
                      np.vstack(self.as_mem[idxs]), \
                      np.vstack(self.rs_mem[idxs]), \
                      np.vstack(self.ps_mem[idxs]), \
                      np.vstack(self.ds_mem[idxs])
        return experiences

    def __len__(self):
        """
        get buffer size
        """
        return self.size
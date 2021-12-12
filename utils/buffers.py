import collections
import numpy as np
import random
import torch

# Replay buffer
class ReplayBuffer:
    
    # create replay buffer of size N
    def __init__(self, N):
        self.buf = collections.deque(maxlen = N)
    
    # add: add a transition (s, a, r, s2, d) or (s, a, r, s2, d, n)
    def add(self, *args):
        self.buf.append(args)
    
    # sample: return minibatch of size n
    def sample(self, n, t):
        minibatch = random.sample(self.buf, n)

        res = [chk for chk in zip(*minibatch)]
        if len(res) == 5:
            S, A, R, S2, D = res
            return t.f(S), t.l(A), t.f(R), t.f(S2), t.i(D)
        elif len(res) == 6:
            S, A, R, S2, D, N = res
            return t.f(S), t.l(A), t.f(R), t.f(S2), t.i(D), t.l(N)
        else:
            assert (False)
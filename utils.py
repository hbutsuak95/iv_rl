import torch
import numpy as np
import random
from collections import namedtuple, deque, Counter

from scipy.optimize import minimize

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_iv_weights(variances):
    '''
    Returns Inverse Variance weights
    Params
    ======
        variances (numpy array): variance of the targets
    '''
    weights = 1/variances
    (weights)
    weights = weights/np.sum(weights)
    (weights)
    return weights

def compute_eff_bs(weights):
    # Compute original effective mini-batch size
    eff_bs = 1/np.sum([weight**2 for weight in weights])
    #print(eff_bs)
    return eff_bs

def get_optimal_xi(variances, minimal_size, epsilon_start):
    minimal_size = min(variances.shape[0] - 1, minimal_size)
    if compute_eff_bs(get_iv_weights(variances)) >= minimal_size:
        return 0        
    fn = lambda x: np.abs(compute_eff_bs(get_iv_weights(variances+np.abs(x))) - minimal_size)
    epsilon = minimize(fn, 0, method='Nelder-Mead', options={'fatol': 1.0, 'maxiter':100})
    xi = np.abs(epsilon.x[0])
    xi = 0 if xi is None else xi
    return xi



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, opt, action_size, seed, device, mask=False):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.opt = opt
        self.action_size = action_size
        self.memory = deque(maxlen=opt.buffer_size)  
        self.batch_size = opt.batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
        self.mask = mask
    
    def add(self, state, action, reward, next_state, done, mask=None):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, sample_size=None):
        """Randomly sample a batch of experiences from memory."""
        if sample_size is None:
            sample_size = self.batch_size

        if sample_size > len(self.memory):
            sample_size = len(self.memory)

        experiences = random.sample(self.memory, k=sample_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        if self.mask:
            if self.opt.mask == "sampling":
                effective_batch_size = self.opt.batch_size*self.opt.mask_prob
                masks = np.zeros((self.opt.batch_size, self.opt.num_nets))
                for i in range(self.opt.num_nets):
                    masks[:effective_batch_size, i] = 1 
                    random.shuffle(masks[:, i]) 
                masks = torch.from_numpy(masks).to(self.device).bool()
            else:
                masks = torch.from_numpy(np.vstack([e.mask for e in experiences if e is not None])).to(self.device).bool()
            return (states, actions, rewards, next_states, dones, masks)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class MaskReplayBuffer(ReplayBuffer):
    def __init__(self, opt, action_size, seed, device):
        super().__init__(opt, action_size, seed, device)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "mask"])

    def add(self, state, action, reward, next_state, done, mask=None):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, mask)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        masks = torch.from_numpy(np.vstack([e.mask for e in experiences if e is not None])).bool().to(self.device)
        return (states, actions, rewards, next_states, dones, masks)


class ReplayBufferBalanced:
        def __init__(self, buffer_size=100000):
                self.obs_risky = None 
                self.next_obs_risky = None
                self.actions_risky = None 
                self.rewards_risky = None 
                self.dones_risky = None
                self.risks_risky = None 
                self.dist_to_fails_risky = None 
                self.costs_risky = None

                self.obs_safe = None 
                self.next_obs_safe = None
                self.actions_safe = None 
                self.rewards_safe = None 
                self.dones_safe = None
                self.risks_safe = None 
                self.dist_to_fails_safe = None 
                self.costs_safe = None

        def add_risky(self, obs, risk):              
                self.obs_risky = obs if self.obs_risky is None else torch.concat([self.obs_risky, obs], axis=0)
                self.risks_risky = risk if self.risks_risky is None else torch.concat([self.risks_risky, risk], axis=0)

        def add_safe(self, obs, risk):
                self.obs_safe = obs if self.obs_safe is None else torch.concat([self.obs_safe, obs], axis=0)
                self.risks_safe = risk if self.risks_safe is None else torch.concat([self.risks_safe, risk], axis=0)

        def sample(self, sample_size):
                idx_risky = range(self.obs_risky.size()[0])
                idx_safe = range(self.obs_safe.size()[0])
                sample_risky_idx = np.random.choice(idx_risky, int(sample_size/2))
                sample_safe_idx = np.random.choice(idx_safe, int(sample_size/2))

                return {"obs": torch.cat([self.obs_risky[sample_risky_idx], self.obs_safe[sample_safe_idx]], 0),
                        "risks": torch.cat([self.risks_risky[sample_risky_idx], self.risks_safe[sample_safe_idx]], 0)}
        



                        

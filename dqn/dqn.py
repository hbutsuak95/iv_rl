import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

import os
import wandb
import random
import numpy as np
from collections import namedtuple, deque, Counter
import wandb
from utils import * 
from .networks import *
 

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, env, opt, device="cuda"):
        """Initialize an Agent object.
        
        Params
        ======
            env (gym object): Initialized gym environment
            opt (dict): command line options for the model
            device (str): cpu or gpu
        """

        self.risk_stats = {}
        import pickle
        if os.path.exists(opt.risk_path):
            self.risk_stats = pickle.load(open(opt.risk_path, "rb"))
            opt.use_risk = True 
            
        self.env = env
        self.opt = opt
        self.state_size = 1 #$env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.seed = random.seed(opt.env_seed)
        self.test_scores = []
        self.device = device
        self.mask = False
        self.risk_size = opt.quantile_num if opt.use_risk else 0 
        # Q-Network
        self.qnetwork_local = QNetwork(self.state_size+self.risk_size, self.action_size, opt.net_seed).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size+self.risk_size, self.action_size, opt.net_seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=opt.lr)

        if os.path.exists(opt.policy_path):
            self.qnetwork_local.load_state_dict(torch.load(opt.policy_path))
            self.qnetwork_target.load_state_dict(torch.load(opt.policy_path))

        # Replay memory
        self.memory = ReplayBuffer(opt, self.action_size, 42, self.device, self.mask)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.xi = 0
        self.loss = 0 


    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if self.mask:
	        mask = self.random_state.binomial(1, self.opt.mask_prob, self.opt.num_nets)
	        self.memory.add(state, action, reward, next_state, done, mask)
        else:
            self.memory.add(state, action, reward, next_state, done)
	        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.opt.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.opt.batch_size:
                experiences = self.memory.sample()
                return self.learn(experiences, self.opt.gamma)
            else:
            	return None

    def act(self, state, eps=0., is_train=False):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        action_values = action_values.cpu().data.numpy()
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values), np.mean(action_values)
        else:
            return random.choice(np.arange(self.action_size)), np.mean(action_values)

    def get_risk(self, state):
        def_risk = [0.1]*10
        try:
            risk = np.histogram(self.risk_stats[obs], range=(0,10), bins=10, density=True)
        except:
            risk = def_risk
        return risk

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        if self.opt.use_risk:
            state_risk = np.array([self.get_risk(states[i]) for i in range(states.size()[0])])
            next_state_risk = np.array([self.get_risk(next_states[i]) for i in range(next_states.size()[0])])
            states = torch.cat([states, torch.Tensor(state_risk).to(self.device)], axis=-1).float()
            next_states = torch.cat([next_states, torch.Tensor(next_state_risk).to(self.device)], axis=-1).float()
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        weights = torch.ones(Q_expected.size()).to(self.device) / self.opt.batch_size
        loss = self.weighted_mse(Q_expected, Q_targets, weights)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # In order to log the loss value
        self.loss = loss.item()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.opt.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def save(self, scores):
        torch.save(self.qnetwork_local.state_dict(), os.path.join(self.opt.log_dir, "%s_%s_seed_%d_net_seed_%d.pth"%(self.opt.env, self.opt.model, self.opt.env_seed, self.opt.net_seed)))
        np.save(os.path.join(self.opt.log_dir,"logs_%s_%s_%s_seed_%d_net_seed_%d.npy"%(self.opt.exp, self.opt.env, self.opt.model, self.opt.env_seed, self.opt.net_seed)), scores)
        wandb.save(os.path.join(self.opt.log_dir,"logs_%s_%s_%s_seed_%d_net_seed_%d.npy"%(self.opt.exp, self.opt.env, self.opt.model, self.opt.env_seed, self.opt.net_seed)), base_path=self.opt.log_dir)


    def weighted_mse(self, inputs, targets, weights, mask = None):
        loss = weights*((targets - inputs)**2)
        if mask is not None:
        	loss *= mask
        #print(loss.size())
        return loss.sum(0)


    def train(self, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """Deep Q-Learning.
        
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        flag = 1 # used for hyperparameter tuning
        scores = []
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        self.risk_stats = {}
        def_risk = [0.1]*10
        ep_obs = []
        num_terminations = 0
        for i_episode in range(1, n_episodes+1):
            obs, _ = self.env.reset()
            if self.opt.use_risk:

                state = np.array([obs] + def_risk)
            else:
                state = np.array([obs])
            score, ep_var, ep_weights, eff_bs_list, xi_list, ep_Q, ep_loss = 0, [], [], [], [], [], []   # list containing scores from each episode
            for t in range(max_t):
                ep_obs.append(obs)
                action, Q = self.act(state, eps, is_train=True)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = np.logical_or(terminated, truncated)
                if self.opt.use_risk:
                    try:
                        risk = np.histogram(self.risk_stats[next_obs], range=(0,10), bins=10, density=True)
                    except:
                        risk = def_risk
                    next_state = np.array([next_obs] + def_risk)
                else:
                    next_state = np.array([next_obs])
                logs = self.step(np.array([obs]), action, reward, np.array([next_obs]), done)
                state = next_state
                obs = next_obs
                if done:
                    reward += self.opt.end_reward
                score += reward
                if logs is not None:
                    # try:
                    ep_var.extend(logs[0])
                    ep_weights.extend(logs[1])
                    eff_bs_list.append(logs[2])
                    xi_list.append(logs[3])
                    # except:
                    #     pass
                ep_Q.append(Q)
                ep_loss.append(self.loss)
                if done:
                    num_terminations += (terminated and reward == 0)
                    if self.opt.use_risk:
                        # print(ep_obs, )
                        e_risks = list(reversed(range(t+1))) if terminated and reward == 0 else [t+1]*(t+1)
                        # print(e_risks)
                        for i in range(t+1):
                            try:
                                self.risk_stats[ep_obs[i]].append(e_risks[i])
                            except:
                                self.risk_stats[ep_obs[i]] = [e_risks[i]]
                        ep_obs = []
                    break 

            #wandb.log({"V(s) (VAR)": np.var(ep_Q), "V(s) (Mean)": np.mean(ep_Q),
            #    "V(s) (Min)": np.min(ep_Q), "V(s) (Max)": np.max(ep_Q), 
            #    "V(s) (Median)": np.median(ep_Q)}, commit=False)
            #wandb.log({"Loss (VAR)": np.var(ep_loss), "Loss (Mean)": np.mean(ep_loss),
            #    "Loss (Min)": np.min(ep_loss), "Loss (Max)": np.max(ep_loss), 
            #    "Loss (Median)": np.median(ep_loss)}, commit=False)
            #if len(ep_var) > 0: # if there are entries in the variance list
	    #        self.train_log(ep_var, ep_weights, eff_bs_list, eps_list)
            # if i_episode % self.opt.test_every == 0:
            #     self.test(episode=i_episode)
 
            scores_window.append(score)        # save most recent score
            scores.append(score)               # save most recent score
            eps = max(eps_end, eps_decay*eps)  # decrease epsilon
            wandb.log({"Moving Average Return/100episode": np.mean(scores_window)}, i_episode)
            wandb.log({"Num terminations ": num_terminations}, i_episode)
            wandb.log({"Episode": i_episode-1}, i_episode)

            #if np.mean(self.test_scores[-100:]) >= self.opt.goal_score and flag:
            #    flag = 0 
            #    wandb.log({"EpisodeSolved": i_episode}, commit=False)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            #self.save(scores)
                
        torch.save(self.qnetwork_local.state_dict(), "qnet_frozenlake.pt")
        if self.opt.use_risk:
            import pickle
            with open("risk_stats_frozen_lake.pkl", "wb") as f:
                pickle.dump(self.risk_stats, f)

    def test(self, episode, num_trials=5, max_t=1000):
        score_list, variance_list = [], []
        #for i in range(num_trials):
        state = self.env.reset()
        score = 0
        for t in range(max_t):
            action, _ = self.act(state, -1)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            score += reward
            if done:
                break
        self.test_scores.append(score)
        #wandb.log({"Test Environment (Moving Average Return/100 episodes)": np.mean(self.test_scores[-100:]),
        #           "Test Environment Return": score}, step=episode)
        return np.mean(score_list), np.var(score_list)


class C51(DQNAgent):
    def __init__(self, env, opt, device="cuda"):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_nets (int): number of Q-networks
            seed (int): random seed
        """
        super().__init__(env, opt, device)

        # Q-Network
        self.v_min = -50
        self.v_max = 50
        self.n_atoms = 51
        self.state_size += self.risk_size
        self.qnetwork_local = c51QNetwork(self.state_size, self.action_size, opt.net_seed, n_atoms=51, v_min=-50, v_max=50).to(self.device)
        self.qnetwork_target = c51QNetwork(self.state_size, self.action_size, opt.net_seed, n_atoms=51, v_min=-50, v_max=50).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=opt.lr)

        if os.path.exists(opt.policy_path):
            self.qnetwork_local.load_state_dict(torch.load(opt.policy_path))
            self.qnetwork_target.load_state_dict(torch.load(opt.policy_path))



    def act(self, state, eps=0., is_train=False):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action, _ = self.qnetwork_local(state)
        self.qnetwork_local.train()
        # action_values = action_values.cpu().data.numpy()
        # Epsilon-greedy action selection
        if random.random() > eps:
            return action, 0 #np.mean(action_values)
        else:
            return random.choice(np.arange(self.action_size)), 0 #np.mean(action_values)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        if self.opt.use_risk:
            state_risk = np.array([self.get_risk(states[i]) for i in range(states.size()[0])])
            next_state_risk = np.array([self.get_risk(next_states[i]) for i in range(next_states.size()[0])])
            states = torch.cat([states, torch.Tensor(state_risk).to(self.device)], axis=-1).float()
            next_states = torch.cat([next_states, torch.Tensor(next_state_risk).to(self.device)], axis=-1).float()
        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            _, next_pmfs = self.qnetwork_target(next_states)
            # Compute Q targets for current states 
            next_atoms = rewards + self.opt.gamma * self.qnetwork_target.atoms * (1 - dones)
            delta_z = self.qnetwork_target.atoms[1] - self.qnetwork_target.atoms[0]
            tz = next_atoms.clamp(self.v_min, self.v_max)

            b = (tz - self.v_min) / delta_z
            l = b.floor().clamp(0, self.n_atoms - 1)
            u = b.ceil().clamp(0, self.n_atoms - 1)

            d_m_l = (u + (l == u).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        _, old_pmfs = self.qnetwork_local(states, actions.flatten())
        loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # In order to log the loss value
        self.loss = loss.item()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.opt.tau)  



class IntrinsicFear(DQNAgent):

    def __init__(self, env, opt, device="cuda"):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_nets (int): number of Q-networks
            seed (int): random seed
        """
        super().__init__(env, opt, device)

        # Q-Network
        self.fear_network = FearNet(self.state_size)
        self.fear_opt = optim.Adam(self.fear_network.parameters(), lr=opt.lr)
        self.fear_criterion = nn.BCEWithLogitsLoss()
        self.fear_rb = ReplayBufferBalanced()
        self.fear_lambda = opt.fear_lambda
        self.fear_network.eval()

    def get_fear(self, state):
        def_risk = [0.1]*10
        pos = list(zip(*np.where(state == 2)))[0][0]
        try:
            risk = (self.risk_stats[pos] < self.fear_radius) / len(self.risk_stats[pos])
        except:
            risk = 0
        return risk
    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        with torch.no_grad():
            # print(Q_targets_next.size(), self.fear_network(next_states)[:, 1].size())
            Q_targets_next = Q_targets_next - self.fear_lambda * self.fear_network(next_states)[:, 1].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        weights = torch.ones(Q_expected.size()).to(self.device) / self.opt.batch_size
        loss = self.weighted_mse(Q_expected, Q_targets, weights)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # In order to log the loss value
        self.loss = loss.item()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.opt.tau)        

    def update_fear_model(self):
        self.fear_network.train()
        data = self.fear_rb.sample(self.opt.batch_size)
        pred = self.fear_network(data["obs"])
       
        loss = self.fear_criterion(pred, torch.nn.functional.one_hot(data["risks"].to(torch.int64), 2).squeeze().float())
        self.fear_opt.zero_grad()
        loss.backward()
        self.fear_opt.step()
        self.fear_network.eval()


    def train(self, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """Deep Q-Learning.
        
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        flag = 1 # used for hyperparameter tuning
        scores = []
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        def_risk = [0.1]*10
        ep_obs = []
        num_terminations = 0
        for i_episode in range(1, n_episodes+1):
            obs, _ = self.env.reset()
            if self.opt.use_risk:

                state = np.array([obs] + def_risk)
            else:
                state = np.array([obs])
            self.fear_lambda = self.opt.fear_lambda * eps
            f_next_obs = None
            score, ep_var, ep_weights, eff_bs_list, xi_list, ep_Q, ep_loss = 0, [], [], [], [], [], []   # list containing scores from each episode
            for t in range(max_t):
                ep_obs.append(obs)
                action, Q = self.act(state, eps, is_train=True)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = np.logical_or(terminated, truncated)
                if self.opt.use_risk:
                    try:
                        risk = np.histogram(self.risk_stats[next_obs], range=(0,10), bins=10, density=True)
                    except:
                        risk = def_risk
                    next_state = np.array([next_obs] + def_risk)
                else:
                    next_state = np.array([next_obs])
                logs = self.step(np.array([obs]), action, reward, np.array([next_obs]), done)
                state = next_state
                obs = next_obs
                f_next_obs = torch.Tensor(next_state).unsqueeze(0) if f_next_obs is None else torch.cat([f_next_obs, torch.Tensor(next_state).unsqueeze(0)])

                if i_episode > 50:
                    self.update_fear_model()

                if done:
                    e_risks = list(reversed(range(t+1))) if  terminated and reward == 0 else [t+1]*(t+1)
                    e_risks = np.array(e_risks)
                    f_risks = torch.Tensor(e_risks)
                    idx_risky = (e_risks<=self.opt.fear_radius)
                    idx_safe = (e_risks>self.opt.fear_radius)
                    risk_ones = torch.ones_like(f_risks)
                    risk_zeros = torch.zeros_like(f_risks)

                    self.fear_rb.add_risky(f_next_obs[idx_risky, :], risk_ones)
                    self.fear_rb.add_safe(f_next_obs[idx_safe, :], risk_zeros)

                if done:
                    reward += self.opt.end_reward
                score += reward
                if logs is not None:
                    # try:
                    ep_var.extend(logs[0])
                    ep_weights.extend(logs[1])
                    eff_bs_list.append(logs[2])
                    xi_list.append(logs[3])
                    # except:
                    #     pass
                ep_Q.append(Q)
                ep_loss.append(self.loss)
                if done:
                    num_terminations += (terminated and reward == 0)
                    if self.opt.use_risk:
                        # print(ep_obs, )
                        e_risks = list(reversed(range(t+1))) if terminated and reward == 0 else [t+1]*(t+1)
                        # print(e_risks)
                        for i in range(t+1):
                            try:
                                self.risk_stats[ep_obs[i]].append(e_risks[i])
                            except:
                                self.risk_stats[ep_obs[i]] = [e_risks[i]]
                        ep_obs = []
                    break 

            #wandb.log({"V(s) (VAR)": np.var(ep_Q), "V(s) (Mean)": np.mean(ep_Q),
            #    "V(s) (Min)": np.min(ep_Q), "V(s) (Max)": np.max(ep_Q), 
            #    "V(s) (Median)": np.median(ep_Q)}, commit=False)
            #wandb.log({"Loss (VAR)": np.var(ep_loss), "Loss (Mean)": np.mean(ep_loss),
            #    "Loss (Min)": np.min(ep_loss), "Loss (Max)": np.max(ep_loss), 
            #    "Loss (Median)": np.median(ep_loss)}, commit=False)
            #if len(ep_var) > 0: # if there are entries in the variance list
	    #        self.train_log(ep_var, ep_weights, eff_bs_list, eps_list)
            # if i_episode % self.opt.test_every == 0:
            #     self.test(episode=i_episode)
 
            scores_window.append(score)        # save most recent score
            scores.append(score)               # save most recent score
            eps = max(eps_end, eps_decay*eps)  # decrease epsilon
            wandb.log({"Moving Average Return/100episode": np.mean(scores_window)}, i_episode)
            wandb.log({"Num terminations ": num_terminations}, i_episode)
            wandb.log({"Episode": i_episode-1}, i_episode)

            #if np.mean(self.test_scores[-100:]) >= self.opt.goal_score and flag:
            #    flag = 0 
            #    wandb.log({"EpisodeSolved": i_episode}, commit=False)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

class LossAttDQN(DQNAgent):

    def __init__(self, env, opt, device="cuda"):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_nets (int): number of Q-networks
            seed (int): random seed
        """
        super().__init__(env, opt, device)

        # Q-Network
        self.qnetwork_local = TwoHeadQNetwork(self.state_size, self.action_size, opt.net_seed).to(self.device)
        self.qnetwork_target = TwoHeadQNetwork(self.state_size, self.action_size, opt.net_seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=opt.lr)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next_all, Q_target_next_var_all = self.qnetwork_target(next_states, True)
        Q_targets_next, next_actions, Q_targets_next_var = Q_targets_next_all.max(1)[0].unsqueeze(1), Q_targets_next_all.max(1)[1].unsqueeze(1),\
        																					Q_target_next_var_all.detach()

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get variance for the next actions
        Q_targets_var = torch.exp(Q_targets_next_var.gather(1, next_actions))

        # Get expected Q values from local model
        Q_expected, Q_log_var  = [x.gather(1, actions) for x in self.qnetwork_local(states, True)] 

        # Compute loss
        self.xi = get_optimal_xi(Q_targets_var.detach().cpu().numpy(), self.opt.minimal_eff_bs, self.xi) if self.opt.dynamic_xi else self.opt.xi
        weights = self.get_mse_weights(Q_targets_var)
        loss = self.weighted_mse(Q_expected, Q_targets, weights)

        # Compute Loss Attenuation 
        y, mu, var = Q_targets, Q_expected, torch.exp(Q_log_var)
        std = torch.sqrt(var) 
        # print(y.size(), mu.size(), std.size())
        lossatt = torch.mean((y - mu)**2 / (2 * torch.square(std)) + (1/2) * torch.log(torch.square(std)))

        net_loss = loss + self.opt.loss_att_weight*lossatt

        # Minimize the loss
        self.optimizer.zero_grad()
        net_loss.backward()
        self.optimizer.step()

        # In order to log the loss value
        self.loss = loss.item()

        eff_batch_size = compute_eff_bs(weights.detach().cpu().numpy())

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.opt.tau)                     

        return torch.exp(Q_log_var).detach().cpu().numpy(), weights.detach().cpu().numpy(), eff_batch_size, self.xi

    def get_mse_weights(self, variance):
    	weights = torch.ones(variance.size()).to(self.device) / self.opt.batch_size
    	return weights

    def train_log(self, var, weights, eff_batch_size, eps_list):
        wandb.log({"Variance(Q) (VAR)": np.var(var), "Variance(Q) (Mean)": np.mean(var),\
        "Variance(Q) (Min)": np.min(var), "Variance(Q) (Max)": np.max(var), "Variance(Q) (Median)": np.median(var)}, commit=False)
        wandb.log({"Variance(Q) (VAR)": np.var(var), "Variance(Q) (Mean)": np.mean(var),
            "Variance(Q) (Min)": np.min(var), "Variance(Q) (Max)": np.max(var), "Variance(Q) (Median)": np.median(var)}, commit=False)
        wandb.log(
            {"Avg Effective Batch Size / Episode": np.mean(eff_batch_size), "Avg Epsilon / Episode": np.mean(eps_list),
            "Max Epsilon / Episode": np.max(eps_list), "Median Epsilon / Episode": np.median(eps_list), 
            "Min Epsilon / Episode": np.min(eps_list)}, commit=False)



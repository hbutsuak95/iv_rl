import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
import pickle
import os
import wandb
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont 

from collections import namedtuple, deque, Counter

from utils import * 
from .networks import *
from island_navigation import *

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
        self.env = env
        self.test_env = IslandNavigationEnvironment(test_env=True)
        self.opt = opt
        if self.opt.safety_info == "gt":
            self.state_size = np.array(env.observation_spec()["board"].shape).prod() + 1
        elif self.opt.safety_info == "emp_risk":
            self.state_size = np.array(env.observation_spec()["board"].shape).prod() + 10
        else:
            self.state_size = np.array(env.observation_spec()["board"].shape).prod()
        self.action_size = env.action_spec().maximum + 1
        self.seed = random.seed(opt.env_seed)
        self.test_scores = []
        self.device = device
        self.mask = False

        # Q-Network
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, opt.net_seed).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, opt.net_seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=opt.lr)

        # Replay memory
        self.memory = ReplayBuffer(opt, self.action_size, 42, self.device, self.mask)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.xi = 0
        self.loss = 0 

        print(env.observation_spec()["board"].shape)
        self.risk_stats = {}
        board_x, board_y = env.observation_spec()["board"].shape[0], env.observation_spec()["board"].shape[1]
        for i in range(board_x*board_y):
            self.risk_stats[i] = list()


    def reset(self):
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.opt.net_seed).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.opt.net_seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.opt.lr)


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
            action_values_t = self.qnetwork_local(state)
        self.qnetwork_local.train()
        action_values = action_values_t.cpu().data.numpy()
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values), np.mean(action_values)
        else:
            return random.choice(np.arange(self.action_size)), np.mean(action_values)

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
        return loss.sum(0)

    def save_board(self, image, save_path, text="Dummy"):
        image = Image.fromarray((image.transpose( 1, 2, 0) / np.max(image) * 255).astype(np.uint8))
        image = image.resize((160, 120), Image.NEAREST)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((10, 10), text, fill="white", font=font)
        image.save(save_path)

        # cv2.imwrite(save_path, (image.transpose( 1, 2, 0) / np.max(image) * 255).astype(np.uint8))
        # plt.imshow(image.transpose(1, 2, 0))
        # plt.axis("off")
        # # plt.text(10, 10, text, fontsize=15)
        # plt.title(text)
        # plt.savefig(save_path, dpi=10)


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
        ep_obs = []
        def_risk = [0.1]*10
        num_terminations = 0
        storage_path = os.path.join("./islandnav/", self.opt.safety_info)
        try:
            os.makedirs(os.path.join(storage_path, "state_visit"))
        except:
            pass
        state_count = np.zeros(48)
        episodes_list = []
        state_count_list = []
        states_repr = []
        q_values = None
        import pandas as pd
        df = pd.DataFrame(columns=["state", "episode", "count"])

        all_states = []
        _, _, _, obs_org = self.env.reset() 
        obs_org = obs_org["board"]
        obs_org[obs_org==2] = 1 # setting the agent's current position to empty space
        # iterating through all positions agent can have.
        safety_values = [1, 2, 3, 2, 1, 1, 2, 3, 2, 1, 1, 2, 3, 3, 2, 1, 1, 2, 3, 2, 1]
        for i, ind in enumerate(np.argwhere(obs_org==1)):
            obs = obs_org.copy()
            obs[ind[0], ind[1]] = 2
            if self.opt.safety_info == "gt":
                obs = list(obs.ravel()) + [safety_values[i]]
            else:
                obs = obs.ravel()
            all_states.append(obs)
        
        # print(all_states)
        all_states = torch.from_numpy(np.array(all_states)).to(self.device).float()
        state_visitations_by_episode = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        recorder_episodes = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        for i_episode in range(1, n_episodes+1):
            # Resetting the network
            if i_episode + 1 % self.opt.reset_freq == 0:
                self.reset()
            state_visit_ep = np.zeros(48)
            level_num = np.random.randint(0, 5)
            self.env = IslandNavigationEnvironment(level_num=level_num)
            _, _, _, old_state = self.env.reset()
            # if i_episode-1 % 10 == 0:
            fig, ax = plt.subplots(1, 2)
            l = ax[1].imshow(state_count.reshape(6, 8), cmap="coolwarm")
            ax[1].axis("off")
            # plt.style.use('ggplot')
            # .savefig(os.path.join(storage_path, "state_visit", "%d.png"%i_episode))
            # plt.close()
            ax[0].imshow(old_state["RGB"].transpose(1, 2, 0))
            ax[0].axis("off")
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(l, cax = cbar_ax)
            fig.text(0.4, 0.8, "Episode = %d"%i_episode)
            fig.savefig(os.path.join(storage_path, "state_visit", "%d.png"%i_episode))
            # fig.close()

            state_count_list.extend(list(state_count))
            episodes_list.extend([i_episode]*48)
            states_repr.extend(list(range(48)))

            if i_episode % 10 == 0 or (i_episode==1):
                q_all_states = self.qnetwork_local(all_states)
                torch.save(q_all_states, "q_all_states_%s_%d.pt"%(self.opt.safety_info, i_episode))



                # self.save_board(state["RGB"], os.path.join(storage_path, "%d_%d.png"%(i_episode, 0)), "Episode=%d | Step=%d"%(i_episode, 0))
            state = old_state["board"].ravel()
            if self.opt.safety_info == "gt":
                safety = self.env.environment_data['safety']
                state = np.array(list(state) + [safety])
            elif self.opt.safety_info == "emp_risk":
                state = np.array(list(state) + def_risk)


            recorder_episodes[level_num].append(i_episode)
            goal_pos = list(zip(*np.where(old_state["board"].ravel() == 4)))[0][0]
            score, ep_var, ep_weights, eff_bs_list, xi_list, ep_Q, ep_loss = 0, [], [], [], [], [], []   # list containing scores from each episode
            for t in range(max_t):
                pos = list(zip(*np.where(old_state["board"].ravel() == 2)))[0][0]
                state_count[pos] += 1
                # if level_num == 0:
                state_visit_ep[pos] += 1

                ep_obs.append(pos)
                action, Q = self.act(state, eps, is_train=True)
                # q_values = Q_a.unsqueeze(0) if q_values is None else torch.cat([q_values, Q_a.unsqueeze(0)], axis=0)
                _, reward, not_done, old_next_state = self.env.step(action)
                if i_episode % 10 == 0:
                    self.save_board(old_next_state["RGB"], os.path.join(storage_path, "%d_%d.png"%(i_episode, t)), "Episode=%d | Step=%d"%(i_episode, t))
                if reward is None or reward < 0:
                    reward = 0
                next_state = old_next_state['board'].ravel()
                if self.opt.safety_info == "gt":
                    safety = self.env.environment_data['safety']
                    next_state = np.array(list(next_state) + [safety])
                elif self.opt.safety_info == "emp_risk":
                    try:
                        risk = np.histogram(self.risk_stats[pos], range=(0,10), bins=10, density=True)
                    except:
                        risk = def_risk
                    next_state = np.array(list(next_state) + def_risk)

             
                
                logs = self.step(state, action, reward, next_state, not not_done)
                state = next_state
                old_state = old_next_state
                if not not_done:
                    if reward > 0:
                        state_visit_ep[goal_pos] += 1
                        
                    e_risks = list(reversed(range(t+1))) if t < max_t-1 else [t]*t
                    for i in range(t+1):
                        self.risk_stats[ep_obs[i]].append(e_risks[i])
                    
                    if (self.env.environment_data['safety'] < 1):
                        reward += self.opt.end_reward
                    ep_obs = []
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
                if not not_done:
                    num_terminations += (self.env.environment_data['safety'] < 1)
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
            # print(state)
            try:
                pos = list(zip(*np.where(old_state["board"].ravel() == 2)))[0][0]
                state_count[pos] += 1
            except:
                pass

            state_visitations_by_episode[level_num].append(state_visit_ep)
            scores_window.append(score)        # save most recent score
            scores.append(score)               # save most recent score
            eps = max(eps_end, eps_decay*eps)  # decrease epsilon
            wandb.log({"Moving Average Return/100episode": np.mean(scores_window)})
            #if np.mean(self.test_scores[-100:]) >= self.opt.goal_score and flag:
            #    flag = 0 
            #    wandb.log({"EpisodeSolved": i_episode}, commit=False)
            wandb.log({"Terminations / Violations": num_terminations})
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            #self.save(scores)
        df["state"] = states_repr
        df["episode"] = episodes_list
        df["count"] = state_count_list
        df.to_csv(os.path.join(storage_path, "stats.csv"), encoding="utf-8")
        with open(os.path.join(storage_path, "state_visitations_%s.pkl"%self.opt.safety_info), "wb") as f:
            pickle.dump(state_count, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open("risk_stats.pkl", "wb") as f:
            pickle.dump(self.risk_stats, f, protocol=pickle.HIGHEST_PROTOCOL)

        state_visitations = {"visit": state_visitations_by_episode, "episodes": recorder_episodes}
        with open(os.path.join(storage_path, "state_visitations_by_episode%s.pkl"%self.opt.safety_info), "wb") as f:
            pickle.dump(state_visitations_by_episode, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(storage_path, "episode_run_by_level%s.pkl"%self.opt.safety_info), "wb") as f:
            pickle.dump(recorder_episodes, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save model at the end 
        torch.save(self.qnetwork_local.state_dict(), "q_net_%s.pt"%self.opt.safety_info)
        # Save qvalues over time 
        # torch.save(q_values, "q_values_%s.pt"%self.opt.safety_info)

    def test(self, episode, num_trials=5, max_t=1000):
        score_list, variance_list = [], []
        #for i in range(num_trials):
        _, _, _, state = self.env.reset()
        state = state["board"].ravel()
        if self.opt.use_safety_info:
            safety = self.test_env.environment_data['safety']
            state = np.array(list(state) + [safety])
        score = 0
        for t in range(max_t):
            action, _ = self.act(state, -1)
            _, reward, not_done, next_state = self.test_env.step(action)
            next_state = next_state["board"].ravel()
            if self.opt.use_safety_info:
                safety = self.test_env.environment_data['safety']
                next_state = np.array(list(next_state) + [safety])
            if reward is None:
                reward = 0
            state = next_state
            score += reward
            if not not_done:
                break
        self.test_scores.append(score)
        wandb.log({"Test Environment (Moving Average Return/100 episodes)": np.mean(self.test_scores[-100:]),
                  "Test Environment Return": score}, step=episode)
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
        self.qnetwork_local = c51QNetwork(self.state_size, self.action_size, opt.net_seed, n_atoms=51, v_min=-50, v_max=50).to(self.device)
        self.qnetwork_target = c51QNetwork(self.state_size, self.action_size, opt.net_seed, n_atoms=51, v_min=-50, v_max=50).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=opt.lr)

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



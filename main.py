import os
import gymnasium as gym
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dqn import * 
# from sac import *
from utils import *
from config import config 
from distutils.util import strtobool

# from codecarbon import EmissionsTracker
# tracker = EmissionsTracker()
# tracker.start()
import wandb
import os 
import time
import warnings
# warnings.filterwarnings('ignore')

# os.environ["WANDB_SILENT"] = "true"

model_dict = {"DQN"                        : DQNAgent,
              "C51"                        : C51,
              # "IntFear"                    : IntrinsicFear,
              "VarDQN"                     : LossAttDQN,
              "EnsembleDQN"                : EnsembleDQN,
            #   "BootstrapDQN"               : RPFMaskEnsembleDQN,
              "IV_EnsembleDQN"             : IV_DQN,              
              "IV_BootstrapDQN"            : IV_BootstrapDQN,
              "BootstrapDQN"               : BootstrapDQN,
              "IV_BootstrapDQN"            : IV_RPFBootstrapDQN,
              "IV_EnsembleDQN"             : IV_DQN,              
              "IV_VarDQN"                  : IV_LossAttDQN,
              "VarEnsembleDQN"             : LakshmiBootstrapDQN,
              "IV_VarEnsembleDQN"          : IV_LakshmiBootstrapDQN,
              "IV_DQN"                     : IV_LakshmiBootstrapDQN,

              "SunriseDQN"                 : Sunrise_BootstrapDQN,
              "Sunrise_VarEnsembleDQN"     : Sunrise_LakshmiBootstrapDQN,

              "UWACDQN"                    : UWAC_DQN,
              "UWAC_VarEnsembleDQN"        : UWAC_LakshmiBootstrapDQN,


            #   "SAC"                        : SACTrainer,
            #   "VarSAC"                     : VarSACTrainer,
            #   "IV_VarSAC"                  : IV_VarSAC,

            #   "EnsembleSAC"                : EnsembleSAC,
            #   "IV_EnsembleSAC"             : IV_EnsembleSAC,
            #   "VarEnsembleSAC"             : VarEnsembleSAC,
            #   "IV_SAC"                     : IV_VarEnsembleSAC,
            #   "IV_VarEnsembleSAC"          : IV_VarEnsembleSAC,

            #   "SunriseSAC"                 : SunriseSAC,
            #   "Sunrise_VarEnsembleSAC"     : Sunrise_VarEnsembleSAC,
              
            #   "UWACSAC"                    : UWACSAC,
            #   "UWAC_VarEnsembleSAC"        : UWAC_VarEnsembleSAC
              }




parser = argparse.ArgumentParser(description="DQN options")
parser.add_argument("--env", type=str, default="LunarLander-v2",
                    help="Gym environment")
parser.add_argument("--map_name", type=str, default="4x4",
                    help="Gym environment")
parser.add_argument("--hole_reward", type=float, default=0,
                    help="Learning rate for SGD update")
parser.add_argument("--policy_path", type=str, default="None",
                    help="Gym environment")
parser.add_argument("--risk_path", type=str, default="None",
                    help="Gym environment")
parser.add_argument("--model", type=str, choices=model_dict.keys(), required=True,
                    help="which RL algorithm to run??")
parser.add_argument("--lr", type=float, default=5e-4,
                    help="Learning rate for SGD update")
parser.add_argument("--batch_size", type=int, default=64,
                   help="batch size")
parser.add_argument("--eff_batch_size", type=int, default=64,
                    help="effective batch size")
parser.add_argument("--buffer_size", type=int, default=int(1e6),
                    help="Replay Buffer Size")
parser.add_argument("--num_nets", type=int, default=5,
                    help="Number of Qnets in the ensemble DQN")
parser.add_argument("--gamma", type=float, default=0.99,
                    help="Discount Factor")
parser.add_argument("--tau", type=float, default=1e-3,
                    help="Target Network Update weight")
parser.add_argument("--mean_target", action="store_true",
                    help="whether to use mean Target for Ensemble Networks")
parser.add_argument("--update_every", type=int, default=1,
                    help="Updating Target Network every x episodes")
parser.add_argument("--log_dir", type=str, default="./logs/",
                    help="location to save models and metadata")
parser.add_argument("--xi", type=float, default=1.,
                    help="xi for variance stabilization")
parser.add_argument("--eps_frac", type=float, default=1.,
                    help="EPS multiplicative factor")
parser.add_argument("--env_seed", type=int, default=0,
                    help="seed for the gym environment")
parser.add_argument("--net_seed", type=int, default=0,
                    help="seed for the neural networks")
parser.add_argument("--num_episodes", type=int, default=210,
                    help="Total number of episodes")
parser.add_argument("--exp", type=str, default="S",
                    help="Experiment category")
parser.add_argument("--tag", type =str, default="",
                    help="Tag for wandb logs")
parser.add_argument("--comment", default="No Comment!",
                    help="add details about what this exp is trying to do ")
parser.add_argument("--mask_prob", default=1.0, type=float,
                    help="how samples are masked to generate diversity across the ensemble")
parser.add_argument("--select_action", default="mean",type=str, choices=["mean", "vote"],
                    help="select action for ensemble based on mean or voting") 
parser.add_argument("--mask", default="bernoulli", type=str, choices=["sampling", "bernoulli"], 
                    help="Sampling a fixed effective batch from a larger batch or using bernoulli masks like BootstrapDQN ")
parser.add_argument("--eps_decay", default=0.99, type=float,
                    help="Exploration decay rate")
parser.add_argument("--dynamic_xi", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="whether to use calculated eps using minimum effective batch size")
parser.add_argument("--eps_type", type=str, choices=["eps", "mebs", "eps_frac"], default="eps",
                    help="how to calculate epsilon for variance stabilization")
parser.add_argument("--minimal_eff_bs", type=float, default=32, 
                    help="Minimal Effective Batch Size for calculating Policy Improvement Noise")
parser.add_argument("--minimal_eff_bs_ratio", type=float, default=1.0, 
                    help="Minimal Effective Batch Ratio for calculating Policy Improvement Noise")
parser.add_argument("--prior_scale", default=10.0, type=float, 
                    help="Prior Scale for RPF ")
parser.add_argument("--loss_att_weight", type=float, default=1.0,
                    help="Weight of loss attenuation in loss function")
parser.add_argument("--test_every", type=int, default=1,
                    help="Test every x episodes")
parser.add_argument("--mcd_prob", type=float, default=0.5,
                    help="Dropout probability for MC Dropout based DQN models")
parser.add_argument("--mcd_samples", type=int, default=5,
                    help="Number of output Q-values to generate using MC Dropout")
parser.add_argument("--same_seed", type=int, default=-1,
                    help="if we want to use same seed for env and net (analysis and sweeps)")
parser.add_argument("--goal_score", type=int,default=200,
                    help="moving average score / 100 at which environment is considered solved")
parser.add_argument("--sunrise_temp", type=float, default=20.0,
                    help="sunrise temperature for weighted Bellman backup")
parser.add_argument("--exploration", type=str, choices=['ts', 'e-greedy', 'bootstrap', 'ucb'], default='e-greedy',
                    help='which exploration technique to use ? (Thomson Sampling(ts), Epsilon-greedy(e-greedy), Bootstrap or UCB exploration')
parser.add_argument("--ucb_lambda", type=float, default=0.0, 
                    help="Lambda to be used for UCB exploration")
parser.add_argument("--end_reward", type=float, default=0.0, 
                    help="Constant added at the end of an episode to shift the return")
parser.add_argument("--burn_in_density", type=int, default=10000,
                    help="Update Density every N steps")
parser.add_argument("--config", default=None,
                    help="configuration to use")
parser.add_argument('--uwac_beta', default=0.5, type=float, 
                    help="beta factor for down-weighing")
parser.add_argument('--clip_bottom', default=0.0, type=float, 
                    help="clip the down-weighing factor by minimum")
parser.add_argument('--clip_top', default=1.5, type=float, 
                    help="clip the down-weighing factor by maximum")
parser.add_argument("--use_exp_weight", type=str2bool, nargs='?',
                    const=True, default=True,
                    help="Use Exponential down-weighing for Q function and/or Policy")
parser.add_argument('--num_sampled_actions', type=int, default=5, 
                    help="Number of actions to sample to calculate variance in the policy")
parser.add_argument("--soft_target_tau", default=5e-3, type=float,
                    help="Soft Target TAU for target network update")
parser.add_argument("--policy_lr", default=3e-4, type=float,
                    help="learning rate for actor updates")
parser.add_argument("--qf_lr", default=3e-4, type=float,
                    help="learning rate for critic updates")
parser.add_argument("--use_bsuite", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="whether to use calculated eps using minimum effective batch size")
# architecture
parser.add_argument('--num_layer', default=2, type=int)
parser.add_argument('--save_freq', default=0, type=int)

parser.add_argument("--use-risk", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    help="Use risk model or not ")
parser.add_argument("--quantile-size", type=int, default=4, help="size of the risk quantile ")
parser.add_argument("--quantile-num", type=int, default=10, help="number of quantiles to make")


parser.add_argument("--fear-radius", default=2, type=int)
parser.add_argument("--fear-lambda", default=1, type=float)
target_type = ["", "_mean_target"]



opt = parser.parse_args()

try:
    config_env = config[opt.env][opt.model]
    for key in config_env.keys():
        setattr(opt, key, config_env[key])
except:
    pass
    
if opt.same_seed >= 0:
    opt.net_seed = opt.same_seed
    opt.env_seed = opt.same_seed



# Inflate batch size when using Masking in Ensemble Methods
# to ensure same effective batch size. 
opt.batch_size = int(opt.eff_batch_size / opt.mask_prob)

opt.minimal_eff_bs = int(opt.minimal_eff_bs_ratio * opt.eff_batch_size)

print(vars(opt))


if "Mean_Target" in opt.model:
    opt.mean_target = True

if __name__ == "__main__":
    device = torch.device("cpu")
    try:
        os.makedirs(opt.log_dir)
    except:
        pass
    wandb.init(config=vars(opt), entity="kaustubh_umontreal",
                    project="risk_aware_exploration",
                    #monitor_gym=True,
                    save_code=True)
                    
    Model = model_dict[opt.model]
    if "sac" not in opt.model.lower():
        env = gym.make(opt.env, map_name=opt.map_name, hole_reward=opt.hole_reward)
        # env.seed(opt.env_seed)
        agent = Model(env, opt, device=device)
        agent.train(n_episodes=opt.num_episodes, eps_decay=opt.eps_decay)
    else:
        run_sac(Model, opt)

# tracker.stop()

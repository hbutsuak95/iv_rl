# IV-RL: Sample Efficient Deep Reinforcement Learning via Uncertainty Estimation 
#### Vincent Mai, Kaustubh Mani and Liam Paull 

#### [[Paper]](https://openreview.net/forum?id=vrW3tvDfOJQ), [[Blog]](https://montrealrobotics.ca/ivrl/)

#### Accepted to [ICLR 2022](https://iclr.cc/)

## Abstract

In model-free deep reinforcement learning (RL) algorithms, using noisy value estimates to supervise policy evaluation and optimization is detrimental to the sample efficiency. As this noise is heteroscedastic, its effects can be mitigated using uncertainty-based weights in the optimization process. Previous methods rely on sampled ensembles, which do not capture all aspects of uncertainty. We provide a systematic analysis of the sources of uncertainty in the noisy supervision that occurs in RL, and introduce inverse-variance RL, a Bayesian framework which combines probabilistic ensembles and Batch Inverse Variance weighting. We propose a method whereby two complementary uncertainty estimation methods account for both the Q-value and the environment stochasticity to better mitigate the negative impacts of noisy supervision. Our results show significant improvement in terms of sample efficiency on discrete and continuous control tasks.

## Installing Dependencies 


	conda create -n iv_rl python=3.7 

	pip install -r requirements.txt


Download [[Mujoco]](https://www.roboti.us/download.html)  (mjpro131) and store it in ~/.mujoco. Also a Mujoco licence key will be needed. 

## Adding Python Paths for Mujoco

	export PYTHONPATH=$PYTHONPATH:$(pwd)/rlkit/
	export PYTHONPATH=$PYTHONPATH:$(pwd)/mbbl_envs/


## Running Mujoco Experiments 

	sh ./scripts/run_mujoco.sh <env_name> <model_name>


env_name can be ["gym_walker2d", "gym_cheetah", "gym_ant", "gym_hopper"]

model_name can be ["SAC", "ProbSAC", "EnsembleSAC", "PredEnsembleSAC", "IV_EnsembleSAC", "IV_ProbEnsembleSAC", "IV_SAC"]


For eg:

	sh ./scripts/run_mujoco.sh gym_walker2d IV_SAC

## Running Bsuite Experiments

	sh ./scripts/run_bsuite.sh cartpole_noise <model_name> 

model_name can be ["DQN", "BootstrapDQN", "ProbEnsembleDQN", "IV_DQN", "IV_ProbEnsembleDQN"]

For eg:
	
	sh ./scripts/run_bsuite.sh cartpole_noise IV_DQN 


## Running Gym Experiments

	sh ./scripts/run_gym.sh <env_name> <model_name>

env_name can be ["LunarLander-v2" or "MountainCar-v0"]

model_name can bet ["DQN", "EnsembleDQN", "ProbDQN", "IV_ProbDQN", "BootstrapDQN", "IV_ProbEnsembleDQN", "IV_DQN", "IV_BootstrapDQN"]


## Acknowledgements

We've used the following repositories to aid our implementation:

- RLkit (https://github.com/rail-berkeley/rlkit)
- MBBL (https://github.com/WilsonWangTHU/mbbl)
- Bsuite (https://github.com/deepmind/bsuite)

## Citation

If you find this work useful, please use the following BibTeX entry for citing us!

```
@article{mai2022sample,
  title={Sample Efficient Deep Reinforcement Learning via Uncertainty Estimation},
  author={Mai, Vincent and Mani, Kaustubh and Paull, Liam},
  journal={arXiv preprint arXiv:2201.01666},
  year={2022}
}
```


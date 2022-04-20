# Multi-armed Bandit Algorithm against Strategic Replication

This repository is the official implementation of ["Multi-armed Bandit Algorithm against Strategic Replication"](https://arxiv.org/abs/2110.12160) accepted by AISTATS 2022.


## Abstract

We consider a multi-armed bandit problem in which a set of arms is registered by each agent, and the agent receives reward when its arm is selected. An agent might strategically submit more arms with replications, which can bring more reward by abusing the bandit algorithm's exploration-exploitation balance. Our analysis reveals that a standard algorithm indeed fails at preventing replication and suffers from linear regret in time T. We aim to design a bandit algorithm which demotivates replications and also achieves a small cumulative regret. We devise Hierarchical UCB (H-UCB) of replication-proof, which has O(lnT)-regret under any equilibrium. We further propose Robust Hierarchical UCB (RH-UCB) which has a sublinear regret even in a realistic scenario with irrational agents replicating careless. We verify our theoretical findings through numerical experiments.

## Code

We implemented Hierarchical UCB and Robust Hierarchical UCB algorithms based on [SMPyBandits](https://github.com/SMPyBandits/SMPyBandits). The implemented algorithms are in *SMPyBandits/Policies/H_UCB.py* and *SMPyBandits/Policies/RH_UCB.py*. And codes for running experiments are in *strategic_scripts/* directory.


### How to Run Experiments

1. Prepare an environment using *docker/Dockerfile*.
2. Make a json file for experiment setup in *strategic_scripts/setups/* directory.
3. Run a python script `python3 main.py --setup={setup_file_name} --policy={policy_name}`, or make a shell script and run it.


### Description of Experiment Json File

```
horizon: The horizon of an experiment.
repetitions: The number of repetitions of an experiment.
n_jobs: The number of jobs for parallelization to accelerate an experiment.
verbosity: A flag to verbose logging.
arm_type: The type of arms used in an experiment.
agent_arm_dict: A dictionary indicating which agent has which arms and how many.
save_json: A flag to save an experiment result into a json file.
save_h5py: A flag to save an experiment result into a h5py file.
save_pickle: A flag to save an experiment result into a pickle file.
```

Example: See *strategic_scripts/setups* directory.


### Argument Description of Running Scripts

```
setup: The location of an experiment setup json file.
policy: A bandit algorithm to use.
L: A hyperparameter to control the number of subsampled arms (Only used by RH-UCB, S-UCB).
```

Example: See shell scripts like *run_h_ucb.sh*.


### How to Experiment with Other Algorithms

1. Make a python file for a new policy.

If you want to use algorithms other than those used in our paper, you can do that by putting your own algorithm into *SMPyBandits/Policies/* directory. Also you can use other popular bandit algorithms like KL-UCB or Thompson Sampling thanks to the [base repository](https://github.com/SMPyBandits/SMPyBandits).

2. Change some parts of code in *strategic_scripts* directory.

Add your policy into `AVALIABLE_POLICY` in *strategic_scripts/main.py* and add your case into `make_env_dict` function in *strategic/help_experiment.py*.


## Cite

Please cite our paper if you use our algorithms or this code in your work:
```
@inproceedings{suho2022band,
  title={Multi-armed Bandit Algorithm against Strategic Replication},
  author={Suho Shin and Seungjoon Lee and Jungseul Ok},
  booktitle=AISTATS,
  year={2022},
  url={https://arxiv.org/abs/2110.12160}
}
```

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings
import os
import sys
import h5py
import pickle
import numpy as np
from datetime import datetime
from pytz import timezone
import math
import random

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    import ujson
except ImportError:
    ujson = None
    import json

from SMPyBandits.Environment import StrategicEvaluator, tqdm
from SMPyBandits.Arms import Bernoulli
from SMPyBandits.Policies import *
from help_experiment import save_experiment_json, flatten, repeat_elem, make_env_dict

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


def run_experiment(
    configuration, L,
    save_json, json_name,
    save_h5py, h5py_name,
    save_pickle, pickle_name
):
    evaluation = StrategicEvaluator(configuration)

    cur_time = datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print(f'\nBefore startOneEnv, current time: {cur_time}\n')

    # Start env and run the experiment
    for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
        evaluation.startOneEnv(envId, env)
    
    print()
    cur_time = datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print(f'\nAfter startOneEnv, current time: {cur_time}\n')

    # Save experiment results
    if L:
        json_name = json_name.replace('.json', f'_L{L}.json')
        h5py_name = h5py_name.replace('.hdf5', f'_L{L}.hdf5')
        pickle_name = pickle_name.replace('.pickle', f'_L{L}.pickle')

    if save_json:
        save_experiment_json(evaluation, json_name)
        print(f'Saved data on {json_name}')
        print()

    if save_h5py:
        evaluation.saveondisk(h5py_name)
        print(f'Saved evaluation on {h5py_name}')
        print()

    if save_pickle:
        with open(pickle_name, 'wb') as picklefile:
            pickle.dump(evaluation, picklefile, pickle.HIGHEST_PROTOCOL)
            print(f'Saved evaluation on {pickle_name}')
            print()


def runner(
        experiment_name, policy_str,
        horizon, repetitions, n_jobs, verbosity,
        arm_type, agent_arm_dict, L,
        save_json, save_h5py, save_pickle
    ):
    print('------------------------- System Check ---------------------------')
    print(f'Available CPU cores: {os.cpu_count()}')
    tot_m, used_m, free_m = map(int, os.popen('free -t -g').readlines()[-1].split()[1:])
    print(f'tot_m: {tot_m}, used_m: {used_m}, free_m: {free_m}')
    print('-----------------------------------------------------------------\n\n')

    print('------------------------- Basic Setup ----------------------------')
    print(f'experiment_name: {experiment_name}, policy_str: {policy_str}')
    print(f'horizon: {horizon}, repetitions: {repetitions}, n_jobs: {n_jobs}, verbosity: {verbosity}')
    print(f'arm_type: {arm_type}, L: {L}, agent_arm_dict: {agent_arm_dict}')
    print(f'save_json: {save_json}, save_h5py:{save_h5py}, save_pickle:{save_pickle}')
    print('------------------------- Basic Setup ----------------------------\n\n')

    if arm_type == "Bernoulli":
        arm_type = Bernoulli
    else:
        raise Exception("Arm other than Bernoulli cannot be run now!")

    environment, policies = make_env_dict(policy_str, agent_arm_dict,
                                          arm_type, horizon, L)

    configuration = {
        "horizon": horizon,
        "repetitions": repetitions,
        "n_jobs": n_jobs,    # = nb of CPU cores
        "verbosity": verbosity,
        "environment": [environment], # = Arms
        "policies": policies, # = Algorithms
        "change_labels": {
            0: policy_str
        }
    }
    print('\n\n************************************ Policy ******************************************************\n')
    print(policies[0])
    print('\n**************************************************************************************************\n\n')

    json_name = os.path.abspath(os.path.join(f'./results/{experiment_name}_{policy_str}.json'))
    h5py_name = os.path.abspath(os.path.join(f'./results/{experiment_name}_{policy_str}.hdf5'))
    pickle_name = os.path.abspath(os.path.join(f'./results/{experiment_name}_{policy_str}.pickle'))

    run_experiment(
        configuration=configuration, L=L,
        save_json=save_json, json_name=json_name,
        save_h5py=save_h5py, h5py_name=h5py_name,
        save_pickle=save_pickle, pickle_name=pickle_name
    )

    print(f'************** Experiment Done! Experiment name: {experiment_name}, Policy name: {policy_str} **************')

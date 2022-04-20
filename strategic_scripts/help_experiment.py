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

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


def save_experiment_json(evaluation, json_name):
    number_of_envs = len(evaluation.envs)
    nbPolicies = evaluation.nbPolicies
    horizon = evaluation.horizon
    repetitions = evaluation.repetitions

    policy_str_list = [np.string_(policy.__cachedstr__).decode('ascii') for policy in evaluation.policies]
    policy_str_list

    total_result_dict = {}

    cur_time = datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print(f'Extracting result from evaluation... time: {cur_time}\n')
    for idx_env in range(number_of_envs):
        env_str = f'env_{idx_env}'
        
        env_arm_str = str(evaluation.envs[idx_env].arms[0])
        env_arm_means = evaluation.envs[idx_env].means
        env_nb_arms = evaluation.envs[idx_env].nbArms
        env_nb_arms_per_agents = evaluation.envs[idx_env].nbArmsPerAgents
        env_nb_agents = len(env_nb_arms_per_agents)

        env_agent_arm_dict = {}
        left_idx = 0
        for agent in range(env_nb_agents):
            limit = env_nb_arms_per_agents[agent]
            param_list = env_arm_means[left_idx: left_idx+limit]
            left_idx += limit

            elem_counter = dict((x, list(param_list).count(x)) for x in set(param_list))
            agent_param_list = list(elem_counter.keys())
            agent_repeat_list = list(elem_counter.values())

            env_agent_arm_dict[agent] = {
                "param": list(map(float, agent_param_list)),
                "repeat": list(map(int, agent_repeat_list)),
            }

        env_result = {}
        for idx_pol, pol_str in enumerate(policy_str_list):
            arm_chosen_mean = evaluation.getMeanArmChosenNb(policyId=idx_pol, envId=idx_env)
            agent_chosen_mean = evaluation.getMeanAgentChosenNb(policyId=idx_pol, envId=idx_env)
            rewards_per_arms_mean = evaluation.getMeanRewardsPerArms(policyId=idx_pol, envId=idx_env)
            rewards_per_agents_mean = evaluation.getMeanRewardsPerAgents(policyId=idx_pol, envId=idx_env)
            cumulated_regret_mean = evaluation.getCumulatedRegret(policyId=idx_pol, envId=idx_env)

            arm_chosen_std = evaluation.getStdArmChosenNb(policyId=idx_pol, envId=idx_env)
            agent_chosen_std = evaluation.getStdAgentChosenNb(policyId=idx_pol, envId=idx_env)
            rewards_per_arms_std = evaluation.getStdRewardsPerArms(policyId=idx_pol, envId=idx_env)
            rewards_per_agents_std = evaluation.getStdRewardsPerAgents(policyId=idx_pol, envId=idx_env)
            cumulated_regret_std = evaluation.getSTDRegret(policyId=idx_pol, envId=idx_env)

            env_result[pol_str] = {
                "armChosenMean": list(map(float, arm_chosen_mean)),
                "agentChosenMean": list(map(float, agent_chosen_mean)),
                "rewardsPerArmsMean": list(map(float, rewards_per_arms_mean)),
                "rewardsPerAgentsMean": list(map(float, rewards_per_agents_mean)),
                "cumulatedRegretMean": list(map(float, cumulated_regret_mean)),

                "armChosenStd": list(map(float, arm_chosen_std)),
                "agentChosenStd": list(map(float, agent_chosen_std)),
                "rewardsPerArmsStd": list(map(float, rewards_per_arms_std)),
                "rewardsPerAgentsStd": list(map(float, rewards_per_agents_std)),
                "cumulatedRegretStd": list(map(float, cumulated_regret_std)),
            }

        total_result_dict[env_str] = {
            "Arm": env_arm_str,
            "Horizon": int(horizon),
            "Repetitions": int(repetitions),
            "AgentArmSettings": env_agent_arm_dict,
            "Result": env_result
        }

    cur_time = datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print(f'Extract Done! time: {cur_time}\n\n')

    if ujson:
        print(f'ujson.dumps ...\n')
        ret_json_str = ujson.dumps(total_result_dict,
                                    indent=4, ensure_ascii=False)
    else:
        print(f'json.dump ...\n')
        ret_json_str = json.dumps(total_result_dict,
                                    indent=4, ensure_ascii=False)
    
    cur_time = datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print(f'\nOpen and write to {json_name}, time: {cur_time}\n\n')
    with open(json_name, 'w', encoding='utf-8') as f:
        f.write(ret_json_str)

    cur_time = datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print(f'Just wrote to json file. Done! {json_name}, time: {cur_time}\n\n')


def flatten(t):
    return [item for sublist in t for item in sublist]


def repeat_elem(original, n_repeat):
    assert len(original) == len(n_repeat)
    ret = []
    for elem, repeat in zip(original, n_repeat):
        temp = [elem] * repeat
        ret.append(temp)
    return ret


def make_env_dict(policy_str, agent_arm_dict,
                  arm_type, horizon, L=None):
    num_agent = len(agent_arm_dict)
    param_list = flatten([agent_arm_dict[i]['param'] for i in agent_arm_dict])
    repeat_list = flatten([agent_arm_dict[i]['repeat'] for i in agent_arm_dict])

    arm_params = flatten(repeat_elem(param_list, repeat_list))
    arms_per_agent = [sum(agent_arm_dict[i]['repeat']) for i in agent_arm_dict]

    origin_arm_num = [len(agent_arm_dict[i]['param']) for i in agent_arm_dict]
    max_origin_arm_num = max(origin_arm_num)

    origin_max_arm = max(arm_params)
    origin_min_arm = min(arm_params)

    if policy_str == 'UCB':
        environment = {
            "arm_type": arm_type,
            "params": arm_params,
            "nbArmsPerAgents": arms_per_agent
        }
        policies = [{
            "archtype": UCB,
            "params": {}
        }]
        return environment, policies

    elif policy_str == 'H_UCB':
        environment = {
            "arm_type": arm_type,
            "params": arm_params,
            "nbArmsPerAgents": arms_per_agent
        }
        policies = [{
            "archtype": H_UCB,
            "params": {
                "nbAgents": num_agent,
                "nbArmsPerAgents": arms_per_agent
            }
        }]
        return environment, policies

    elif policy_str == 'RH_UCB':
        environment = {
            "arm_type": arm_type,
            "params": arm_params,
            "nbArmsPerAgents": arms_per_agent,
            "maxArm": origin_max_arm,
            "minArm": origin_min_arm
        }
        policies = [{
            "archtype": RH_UCB,
            "params": {
                "nbAgents": num_agent,
                "nbArmsPerAgents": arms_per_agent,
                "horizon": horizon,
                "max_origin_arm_num": L if L else max_origin_arm_num
            }
        }]
        return environment, policies

    elif policy_str == 'Sampled_R_UCB':
        environment = {
            "arm_type": arm_type,
            "params": arm_params,
            "nbArmsPerAgents": arms_per_agent,
            "maxArm": origin_max_arm,
            "minArm": origin_min_arm
        }
        policies = [{
            "archtype": Sampled_R_UCB,
            "params": {
                "nbAgents": num_agent,
                "nbArmsPerAgents": arms_per_agent,
                "horizon": horizon,
                "max_origin_arm_num": L if L else sum(origin_arm_num)
            }
        }]
        return environment, policies

    else:
        raise Exception('Undefined policy entered!')

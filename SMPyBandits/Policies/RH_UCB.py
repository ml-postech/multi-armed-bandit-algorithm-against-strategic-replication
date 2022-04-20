# -*- coding: utf-8 -*-
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "SlyJabiru"
__version__ = "0.1"

import random
from math import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!


try:
    from .StrategicIndexPolicy import StrategicIndexPolicy
except ImportError:
    from StrategicIndexPolicy import StrategicIndexPolicy


class RH_UCB(StrategicIndexPolicy):
    def __init__(self, nbArms, nbAgents, nbArmsPerAgents, horizon, max_origin_arm_num,
                 lower=0., amplitude=1.):
        # Play with only array indices of arms!
        self.horizon = horizon
        self.L = max_origin_arm_num
        self.sampledArms = np.full((nbArms), False, dtype=bool)
        # self.validArmsPerAgents = []

        super(RH_UCB, self).__init__(
            nbArms, nbAgents, nbArmsPerAgents,
            lower=lower, amplitude=amplitude
        )
    
    def sampleArms(self):
        self.sampledArms = np.full((self.nbArms), False, dtype=bool)
        armPossession = np.cumsum(self.nbArmsPerAgents) - 1
        for i in range(self.nbAgents):
            i_arm_num = int(min(self.nbArmsPerAgents[i], self.L * np.log(self.horizon)))
            if i == 0:
                arm_index_list_agent_i = list(range(0, armPossession[i]+1))
            else:
                arm_index_list_agent_i = list(range(armPossession[i-1]+1, armPossession[i]+1))
            
            i_sampled_arms = random.sample(arm_index_list_agent_i, i_arm_num)
            # self.validArmsPerAgents.append(i_arm_num)
            self.sampledArms[i_sampled_arms] = True

    def computeAgentIndex(self, agent):
        if self.agentPulls[agent] < 1:
            return float('+inf')
        else:
            return (self.agentRewards[agent] / self.agentPulls[agent]) + sqrt(sqrt(self.t) * log(self.t) / self.agentPulls[agent])

    def computeArmIndex(self, arm):
        if self.sampledArms[arm] == False:
            return float('-inf')
        elif self.armPulls[arm] < 1:
            return float('+inf')
        else:
            armPossession = np.cumsum(self.nbArmsPerAgents) - 1
            temp = (armPossession >= arm)
            agent = np.where(temp)[0][0]
            return (self.armRewards[arm] / self.armPulls[arm]) + sqrt((2 * log(self.agentPulls[agent])) / self.armPulls[arm])

    def computeAllIndex(self):
        """ Compute the current indices for all agent and all arms, in a vectorized manner."""
        agentIndices = (self.agentRewards / self.agentPulls) + np.sqrt(np.sqrt(self.t) * np.log(self.t) / self.agentPulls)

        agentPullsRepeated = np.repeat(self.agentPulls, self.nbArmsPerAgents)
        armIndices = (self.armRewards / self.armPulls) + np.sqrt((2 * np.log(agentPullsRepeated)) / self.armPulls)

        unsampledArms = np.where(~self.sampledArms)[0]

        agentIndices[self.agentPulls < 1] = float('+inf')
        armIndices[self.armPulls < 1] = float('+inf')
        armIndices[unsampledArms] = float('-inf')
        
        self.agentIndex[:] = agentIndices
        self.armIndex[:] = armIndices

    def startGame(self):
        super(RH_UCB, self).startGame()
        self.sampleArms()
        print(f'-------------- Sampled Arms Offsets -----------\n')
        print(f'L: {self.L} (max original arm num)')
        print(f'Len: {len(np.where(self.sampledArms)[0])}')
        print(np.where(self.sampledArms)[0])
        print(f'\n-----------------------------------------------')

# --- Debugging

# if __name__ == "__main__":
#     # Code for debugging purposes.
#     from doctest import testmod
#     print("\nTesting automatically all the docstring written in each functions of this module :")
#     testmod(verbose=True)

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


class Sampled_R_UCB(StrategicIndexPolicy):
    def __init__(self, nbArms, nbAgents, nbArmsPerAgents, horizon, max_origin_arm_num,
                 lower=0., amplitude=1.):
        # Play with only array indices of arms!
        self.horizon = horizon
        self.L = max_origin_arm_num
        self.sampledArms = np.full((nbArms), False, dtype=bool)
        # self.validArmsPerAgents = []

        super(Sampled_R_UCB, self).__init__(
            nbArms, nbAgents, nbArmsPerAgents,
            lower=lower, amplitude=amplitude
        )

    def sampleArms(self):
        self.sampledArms = np.full((self.nbArms), False, dtype=bool)
        arm_num = int(min(self.nbArms, self.L * np.log(self.horizon)))
        index_list = list(range(self.nbArms))
        sampled_arms = random.sample(index_list, arm_num)
        self.sampledArms[sampled_arms] = True

    def computeAgentIndex(self, agent):
        return float('+inf')

    def computeArmIndex(self, arm):
        if self.sampledArms[arm] == False:
            return float('-inf')
        elif self.armPulls[arm] < 1:
            return float('+inf')
        else:
            return (self.armRewards[arm] / self.armPulls[arm]) + sqrt((2 * log(self.t)) / self.armPulls[arm])

    def computeAllIndex(self):
        """ Compute the current indices for all agent and all arms, in a vectorized manner."""
        armIndices = (self.armRewards / self.armPulls) + np.sqrt((2 * np.log(self.t)) / self.armPulls)

        unsampledArms = np.where(~self.sampledArms)[0]

        agentIndices = np.full((self.nbAgents), float('+inf'))
        armIndices[self.armPulls < 1] = float('+inf')
        armIndices[unsampledArms] = float('-inf')

        self.agentIndex[:] = agentIndices
        self.armIndex[:] = armIndices

    def startGame(self):
        super(Sampled_R_UCB, self).startGame()
        self.sampleArms()
        print(f'-------------- Sampled Arms Offsets -----------\n')
        print(f'L: {self.L} (max original arm num)')
        print(f'Len: {len(np.where(self.sampledArms)[0])}')
        print(np.where(self.sampledArms)[0])
        print(f'\n-----------------------------------------------')

    def choice(self):
        self.computeAllIndex()
        # Uniform choice among the best arms
        try:
            return np.random.choice(np.nonzero(self.armIndex == np.max(self.armIndex))[0])
        except ValueError:
            print("Warning: unknown error in Sampled_R_UCB.choice(): the indexes were {} but couldn't be used to select an arm.".format(self.armIndex))
            return np.random.randint(self.nbArms)

# --- Debugging

# if __name__ == "__main__":
#     # Code for debugging purposes.
#     from doctest import testmod
#     print("\nTesting automatically all the docstring written in each functions of this module :")
#     testmod(verbose=True)

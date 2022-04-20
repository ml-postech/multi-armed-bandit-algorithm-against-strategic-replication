# -*- coding: utf-8 -*-
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "SlyJabiru"
__version__ = "0.1"

import random
from math import sqrt, log, pow
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!


try:
    from .StrategicIndexPolicy import StrategicIndexPolicy
except ImportError:
    from StrategicIndexPolicy import StrategicIndexPolicy


class PRH_UCB(StrategicIndexPolicy):
    def __init__(self, nbArms, nbAgents, nbArmsPerAgents,
                 lower=0., amplitude=1.):
        # Play with only array indices of arms!
        self.sampledArms = np.full((nbArms), False, dtype=bool)
        self.nbSampledArmPerAgents = np.zeros((nbAgents), dtype=int)
        super(PRH_UCB, self).__init__(nbArms, nbAgents, nbArmsPerAgents,
                                     lower=lower, amplitude=amplitude)
    
    def computeAgentIndex(self, agent):
        if self.agentPulls[agent] < 1:
            return float('+inf')
        else:
            return (self.agentRewards[agent] / self.agentPulls[agent]) + sqrt(sqrt(self.t * pow(log(self.t), 3)) / self.agentPulls[agent])

    def computeArmIndex(self, arm):
        armPossession = np.cumsum(self.nbArmsPerAgents) - 1
        temp = (armPossession >= arm)
        agent = np.where(temp)[0][0]

        threshold = np.minimum(self.nbArmsPerAgents[agent], np.power(np.log(self.t), 2))
        cond = self.nbSampledArmPerAgents[agent] < threshold

        if np.bool(cond) and (self.armPulls[arm] < 1):
            # Should be explored. Arm pool should be expanded
            return  float('+inf')
        elif not np.bool(cond) and (self.armPulls[arm] < 1):
            # Should be excluded. Arm pool should be holded.
            return  float('-inf')
        else:
            return (self.armRewards[arm] / self.armPulls[arm]) + sqrt((2 * log(self.agentPulls[agent])) / self.armPulls[arm])

    def computeAllIndex(self):
        """ Compute the current indices for all agent and all arms, in a vectorized manner."""
        agentIndices = (self.agentRewards / self.agentPulls) + np.sqrt(np.sqrt(self.t * np.power(np.log(self.t),3)) / self.agentPulls)

        agentPullsRepeated = np.repeat(self.agentPulls, self.nbArmsPerAgents)
        armIndices = (self.armRewards / self.armPulls) + np.sqrt((2 * np.log(agentPullsRepeated)) / self.armPulls)

        threshold = np.minimum(self.nbArmsPerAgents, np.power(np.log(self.t), 2))
        cond = self.nbSampledArmPerAgents < threshold
        condRepeated = np.repeat(cond, self.nbArmsPerAgents)

        agentIndices[self.agentPulls < 1] = float('+inf')

        # if (cond) => The pool of arm should be increased -> +inf for unexplored arms
        # else      => The pool is full -> -inf for unexplored arms, already full, those arms should not be pulled
        armIndices[condRepeated & (self.armPulls < 1)] = float('+inf')
        armIndices[~condRepeated & (self.armPulls < 1)] = float('-inf')
        
        self.agentIndex[:] = agentIndices
        self.armIndex[:] = armIndices
        
    def choice(self):
        self.computeAllIndex()

        # Uniform choice among the best agents
        try:
            agent = np.random.choice(np.nonzero(self.agentIndex == np.max(self.agentIndex))[0])
        except ValueError:
            print("Warning: unknown error in PRH_UCB.choice(): the agent indexes were {} but couldn't be used to select an agent.".format(self.agentIndex))
            agent = np.random.randint(self.nbAgents)

        # Get arm indices with chosen agent
        tempArmIndexArr = np.full((self.nbArms), float('-inf'))
        armPossession = np.cumsum(self.nbArmsPerAgents) - 1

        if agent == 0:
            tempArmIndexArr[0:armPossession[agent]+1] = self.armIndex[0:armPossession[agent]+1]
        else:
            tempArmIndexArr[armPossession[agent-1]+1:armPossession[agent]+1] = self.armIndex[armPossession[agent-1]+1:armPossession[agent]+1]

        # Uniform choice among the best arms of the best agent.
        # np.nonzero gives an index
        try:
            arm = np.random.choice(np.nonzero(tempArmIndexArr == np.max(tempArmIndexArr))[0])
            self.nbSampledArmPerAgents[agent] += (~self.sampledArms[arm])
            self.sampledArms[arm] = True
            return arm
        except ValueError:
            print("Warning: unknown error in PRH_UCB.choice(): the arm indexes were {} but couldn't be used to select an arm.".format(self.armIndex))
            return np.random.choice(np.where(tempArmIndexArr >= 0)[0])



# --- Debugging

# if __name__ == "__main__":
#     # Code for debugging purposes.
#     from doctest import testmod
#     print("\nTesting automatically all the docstring written in each functions of this module :")
#     testmod(verbose=True)

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


class RH_UCB_Temp(StrategicIndexPolicy):
    def __init__(self, nbArms, nbAgents, nbArmsPerAgents,
                 lower=0., amplitude=1.):
        super(RH_UCB_Temp, self).__init__(nbArms, nbAgents, nbArmsPerAgents,
                                     lower=lower, amplitude=amplitude)

    def computeAgentIndex(self, agent):
        if self.agentPulls[agent] < 1:
            return float('+inf')
        else:
            return (self.agentRewards[agent] / self.agentPulls[agent]) + sqrt(sqrt(self.t) * log(self.t) / self.agentPulls[agent])

    def computeArmIndex(self, arm):
        if self.armPulls[arm] < 1:
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

        agentIndices[self.agentPulls < 1] = float('+inf')
        armIndices[self.armPulls < 1] = float('+inf')

        self.agentIndex[:] = agentIndices
        self.armIndex[:] = armIndices


# --- Debugging

# if __name__ == "__main__":
#     # Code for debugging purposes.
#     from doctest import testmod
#     print("\nTesting automatically all the docstring written in each functions of this module :")
#     testmod(verbose=True)

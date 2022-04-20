# -*- coding: utf-8 -*-
""" Base class for any policy.

- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3], lower = -3, amplitude = 6.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "SlyJabiru"
__version__ = "0.1"

import numpy as np

#: If True, every time a reward is received, a warning message is displayed if it lies outsides of ``[lower, lower + amplitude]``.
CHECKBOUNDS = True
CHECKBOUNDS = False


class StrategicBasePolicy(object):
    """ Base class for any policy."""

    def __init__(self, nbArms, nbAgents, nbArmsPerAgents,
                 lower=0., amplitude=1.):
        """ New policy."""
        # Parameters
        assert nbArms > 0, "Error: the 'nbArms' parameter of a {} object cannot be <= 0.".format(self)  # DEBUG
        assert nbAgents > 0, "Error: the 'nbAgents' parameter of a {} object cannot be <= 0.".format(self)  # DEBUG
        assert len(nbArmsPerAgents) == nbAgents, \
            f"Error: len(nbArmsPerAgents) != nbPlayers, len(nbArmsPerAgents): {len(nbArmsPerAgents)}, nbPlayers: {nbAgents}"
        assert sum(nbArmsPerAgents) == nbArms, \
            f"Error: sum(nbArmsPerAgents) != nbArms, sum(nbArmsPerAgents): {sum(nbArmsPerAgents)}, nbArms: {nbArms}"
        assert amplitude > 0, "Error: the 'amplitude' parameter of a {} object cannot be <= 0.".format(self)  # DEBUG

        self.nbArms = nbArms  #: Number of arms
        self.nbAgents = nbAgents  # Number of agents
        self.nbArmsPerAgents = np.array(nbArmsPerAgents)  # List of the number of arms, for each agent
        self.lower = lower  #: Lower values for rewards
        self.amplitude = amplitude  #: Larger values for rewards

        # Internal memory
        self.t = 0  #: Internal time
        self.armPulls = np.zeros(nbArms, dtype=int)  #: Number of pulls of each arms
        self.armRewards = np.zeros(nbArms)  #: Cumulated rewards of each arms
        self.agentPulls = np.zeros(nbAgents,
                                   dtype=int)  #: Number of pulls of each agent. Should be the sum of pulls of arms corresponding agent
        self.agentRewards = np.zeros(
            nbAgents)  #: Cumulated rewards of each agent. Should be the sum of rewards of arms corresponding agent

    def __str__(self):
        """ -> str"""
        return self.__class__.__name__

    # --- Start game, and receive rewards

    def startGame(self):
        """ Start the game (fill pulls and rewards with 0)."""
        self.t = 0
        self.agentPulls.fill(0)
        self.agentRewards.fill(0)
        self.armPulls.fill(0)
        self.armRewards.fill(0)


    if CHECKBOUNDS:
        # XXX useless checkBounds feature
        def getReward(self, arm, reward):
            """ Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1])."""
            assert arm < self.nbArms, f"Arm index is greater or equal than total number of arms. arm: {arm}, nbArms: {self.nbArms}"
            
            armPossession = np.cumsum(self.nbArmsPerAgents) - 1
            temp = (armPossession >= arm)
            agent = np.where(temp)[0][0]
            
            self.t += 1
            self.agentPulls[agent] += 1
            self.armPulls[arm] += 1
            
            ########################################################################################################
            # XXX we could check here if the reward is outside the bounds
            if not 0 <= reward - self.lower <= self.amplitude:
                print(
                    "Warning: {} received on arm {} a reward = {:.3g} that is outside the interval [{:.3g}, {:.3g}] : the policy will probably fail to work correctly...".format(
                        self, arm, reward, self.lower, self.lower + self.amplitude))  # DEBUG
            # else:
            #     print("Info: {} received on arm {} a reward = {:.3g} that is inside the interval [{:.3g}, {:.3g}]".format(self, arm, reward, self.lower, self.lower + self.amplitude))  # DEBUG
            ########################################################################################################
            
            reward = (reward - self.lower) / self.amplitude
            self.agentRewards[agent] += reward
            self.armRewards[arm] += reward
    else:
        # It's faster to define two methods and pick one
        # (one test in init, that's it)
        # rather than doing the test in the method
        def getReward(self, arm, reward):
            """ Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1])."""
            
            assert arm < self.nbArms, f"Arm index is greater or equal than total number of arms. arm: {arm}, nbArms: {self.nbArms}"
            
            armPossession = np.cumsum(self.nbArmsPerAgents) - 1
            temp = (armPossession >= arm)
            agent = np.where(temp)[0][0]
            
            self.t += 1
            self.agentPulls[agent] += 1
            self.armPulls[arm] += 1
                        
            reward = (reward - self.lower) / self.amplitude
            self.agentRewards[agent] += reward
            self.armRewards[arm] += reward

            
    # --- Basic choice() and handleCollision() method

    def choice(self):
        """ Not defined."""
        raise NotImplementedError(
            "This method choice() has to be implemented in the child class inheriting from StrategicBasePolicy.")

    # def handleCollision(self, arm, reward=None):
    #     """ Default to give a 0 reward (or ``self.lower``)."""
    #     # print("DEBUG StrategicBasePolicy.handleCollision({}, {}) was called...".format(arm, reward))  # DEBUG
    #     # self.getReward(arm, self.lower if reward is None else reward)
    #     self.getReward(arm, self.lower)
    #     # raise NotImplementedError("This method handleCollision() has to be implemented in the child class inheriting from StrategicBasePolicy.")

    # --- Others choice...() methods, partly implemented

    def choiceWithRank(self, rank=1):
        """ Not defined."""
        if rank == 1:
            return self.choice()
        else:
            raise NotImplementedError(
                "This method choiceWithRank(rank) has to be implemented in the child class inheriting from StrategicBasePolicy.")

    def choiceFromSubSet(self, availableArms='all'):
        """ Not defined."""
        if availableArms == 'all':
            return self.choice()
        else:
            raise NotImplementedError(
                "This method choiceFromSubSet(availableArms) has to be implemented in the child class inheriting from StrategicBasePolicy.")

    def choiceMultiple(self, nb=1):
        """ Not defined."""
        if nb == 1:
            return np.array([self.choice()])
        else:
            raise NotImplementedError(
                "This method choiceMultiple(nb) has to be implemented in the child class inheriting from StrategicBasePolicy.")

    def choiceIMP(self, nb=1, startWithChoiceMultiple=True):
        """ Not defined."""
        if nb == 1:
            return np.array([self.choice()])
        else:
            return self.choiceMultiple(nb=nb)

#     def estimatedOrder(self):
#         """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing means.

#         - For a base policy, it is completely random.
#         """
#         return np.random.permutation(self.nbArms)

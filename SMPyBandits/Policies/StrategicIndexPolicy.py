# -*- coding: utf-8 -*-
""" Generic index policy.

- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3], lower = -3, amplitude = 6.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "SlyJabiru"
__version__ = "0.1"

import numpy as np

try:
    from .StrategicBasePolicy import StrategicBasePolicy
except (ImportError, SystemError):
    from StrategicBasePolicy import StrategicBasePolicy


class StrategicIndexPolicy(StrategicBasePolicy):
    """ Class that implements a generic index policy."""

    def __init__(self, nbArms, nbAgents, nbArmsPerAgents,
                 lower=0., amplitude=1.):
        """ New index policy for strategic bandit.

        - nbArms: the number of arms,
        - nbAgents: the number of agents,
        - nbArmsPerAgents: a list of the number of arms for each agent
        - lower, amplitude: lower value and known amplitude of the rewards.
        """
        super(StrategicIndexPolicy, self).__init__(nbArms, nbAgents, nbArmsPerAgents,
                                                   lower=lower, amplitude=amplitude)
        self.agentIndex = np.zeros(nbAgents)  #: Numerical index for each agents
        self.armIndex = np.zeros(nbArms)  #: Numerical index for each arms

    # --- Start game, and receive rewards

    def startGame(self):
        """ Initialize the policy for a new game."""
        super(StrategicIndexPolicy, self).startGame()
        self.agentIndex.fill(0)
        self.armIndex.fill(0)

    
    def computeAgentIndex(self, agent):
        """ Compute the current index of agent 'argent'."""
        raise NotImplementedError("This method computeAgentIndex(agent) has to be implemented in the child class inheriting from StrategicIndexPolicy.")


    def computeArmIndex(self, arm):
        """ Compute the current index of arm 'arm'."""
        raise NotImplementedError("This method computeArmIndex(arm) has to be implemented in the child class inheriting from StrategicIndexPolicy.")


    def computeAllIndex(self):
        """ Compute the current indexes for all arms. Possibly vectorized, by default it can *not* be vectorized automatically."""
        for agent in range(self.nbAgents):
            self.agentIndex[agent] = self.computeAgentIndex(agent)
        
        for arm in range(self.nbArms):
            self.armIndex[arm] = self.computeArmIndex(arm)


    def choice(self):
        r""" In an strategic index policy,
        choose an arm with maximal index of the maximal index agent (uniformly at random):

        .. warning:: In almost all cases, there is a unique arm with maximal index, so we loose a lot of time with this generic code, but I couldn't find a way to be more efficient without loosing generality.
        """
        # I prefer to let this be another method, so child of IndexPolicy only needs to implement it (if they want, or just computeIndex)
        self.computeAllIndex()

        # Uniform choice among the best agents
        try:
            agent = np.random.choice(np.nonzero(self.agentIndex == np.max(self.agentIndex))[0])
        except ValueError:
            print("Warning: unknown error in StrategicIndexPolicy.choice(): the agent indexes were {} but couldn't be used to select an agent.".format(self.agentIndex))
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
            return np.random.choice(np.nonzero(tempArmIndexArr == np.max(tempArmIndexArr))[0])
        except ValueError:
            print("Warning: unknown error in StrategicIndexPolicy.choice(): the arm indexes were {} but couldn't be used to select an arm.".format(self.armIndex))
            return np.random.choice(np.where(tempArmIndexArr >= 0)[0])


#     # TODO: Evaluator.py uses estimatedOrder. However, it is not necessary I think.
#     # TODO: compute agents, arms
#     def estimatedOrder(self):
#         """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing means."""
#         self.computeAllIndex()
#         return np.argsort(self.index)


#     # Maybe this is not needed
#     def estimatedBestArms(self, M=1):
#         """ Return a (non-necessarily sorted) list of the indexes of the M-best arms. Identify the set M-best."""
#         assert 1 <= M <= self.nbArms, "Error: the parameter 'M' has to be between 1 and K = {}, but it was {} ...".format(self.nbArms, M)  # DEBUG
#         # # WARNING this slows down everything, but maybe the only way to make this correct?
#         # if np.all(np.isinf(self.index)):
#         #     # Initial guess: random estimate of the set Mbest
#         #     choice = np.random.choice(self.nbArms, size=M, replace=False)
#         #     print("Warning: estimatedBestArms() for self = {} was called with M = {} but all indexes are +inf, so using a random estimate = {} of Mbest instead of the biased [K-M,...,K-1] ...".format(self, M, choice))  # DEBUG
#         #     return choice
#         # else:
#         order = self.estimatedOrder()
#         return order[-M:]

        
    # --- Others choice...() methods


#     # TODO: compute agents, arms
#     def choiceWithRank(self, rank=1):
#         """ In an index policy, choose an arm with index is the (1+rank)-th best (uniformly at random).

#         - For instance, if rank is 1, the best arm is chosen (the 1-st best).
#         - If rank is 4, the 4-th best arm is chosen.


#         .. note:: This method is *required* for the :class:`PoliciesMultiPlayers.rhoRand` policy.

#         """
#         if rank == 1:
#             return self.choice()
#         else:
#             assert rank >= 1, "Error: for IndexPolicy = {}, in choiceWithRank(rank={}) rank has to be >= 1.".format(self, rank)
#             self.computeAllIndex()
#             sortedRewards = np.sort(self.index)
#             # Question: What happens here if two arms has the same index, being the max?
#             # Then it is fair to chose a random arm with best index, instead of aiming at an arm with index being ranked rank
#             chosenIndex = sortedRewards[-rank]
#             # Uniform choice among the rank-th best arms
#             try:
#                 return np.random.choice(np.nonzero(self.index == chosenIndex)[0])
#             except ValueError:
#                 print("Warning: unknown error in IndexPolicy.choiceWithRank(): the indexes were {} but couldn't be used to select an arm.".format(self.index))
#                 return np.random.randint(self.nbArms)


#     # TODO: compute agents, arms
#     def choiceFromSubSet(self, availableArms='all'):
#         """ In an index policy, choose the best arm from sub-set availableArms (uniformly at random)."""
#         if isinstance(availableArms, str) and availableArms == 'all':
#             return self.choice()
#         # If availableArms are all arms? XXX no this could loop, better do it here
#         # elif len(availableArms) == self.nbArms:
#         #     return self.choice()
#         elif len(availableArms) == 0:
#             print("WARNING: IndexPolicy.choiceFromSubSet({}): the argument availableArms of type {} should not be empty.".format(availableArms, type(availableArms)))  # DEBUG
#             # WARNING if no arms are tagged as available, what to do ? choose an arm at random, or call choice() as if available == 'all'
#             return self.choice()
#         else:
#             for arm in availableArms:
#                 self.index[arm] = self.computeIndex(arm)
#             # Uniform choice among the best arms
#             try:
#                 return availableArms[np.random.choice(np.nonzero(self.index[availableArms] == np.max(self.index[availableArms]))[0])]
#             except ValueError:
#                 return np.random.choice(availableArms)


#     # TODO: compute agents, arms
#     def choiceMultiple(self, nb=1):
#         """ In an index policy, choose nb arms with maximal indexes (uniformly at random)."""
#         if nb == 1:
#             return np.array([self.choice()])
#         else:
#             self.computeAllIndex()
#             sortedIndexes = np.sort(self.index)
#             # Uniform choice of nb different arms among the best arms
#             # FIXED sort it then apply affectation_order, to fix its order ==> will have a fixed nb of switches for CentralizedMultiplePlay
#             try:
#                 return np.random.choice(np.nonzero(self.index >= sortedIndexes[-nb])[0], size=nb, replace=False)
#             except ValueError:
#                 return np.random.choice(self.nbArms, size=nb, replace=False)


#     # TODO: compute agents, arms
#     def choiceIMP(self, nb=1, startWithChoiceMultiple=True):
#         """ In an index policy, the IMP strategy is hybrid: choose nb-1 arms with maximal empirical averages, then 1 arm with maximal index. Cf. algorithm IMP-TS [Komiyama, Honda, Nakagawa, 2016, arXiv 1506.00779]."""
#         if nb == 1:
#             return np.array([self.choice()])
#         else:
#             # For first exploration steps, do pure exploration
#             if startWithChoiceMultiple:
#                 if np.min(self.pulls) < 1:
#                     return self.choiceMultiple(nb=nb)
#                 else:
#                     empiricalMeans = self.rewards / self.pulls
#             else:
#                 empiricalMeans = self.rewards / self.pulls
#                 empiricalMeans[self.pulls < 1] = float('inf')
#             # First choose nb-1 arms, from rewards
#             sortedEmpiricalMeans = np.sort(empiricalMeans)
#             exploitations = np.random.choice(np.nonzero(empiricalMeans >= sortedEmpiricalMeans[-nb])[0], size=nb - 1, replace=False)
#             # Then choose 1 arm, from index now
#             availableArms = np.setdiff1d(np.arange(self.nbArms), exploitations)
#             exploration = self.choiceFromSubSet(availableArms)
#             # Affect a random location to is exploratory arm
#             return np.insert(exploitations, np.random.randint(np.size(exploitations) + 1), exploration)

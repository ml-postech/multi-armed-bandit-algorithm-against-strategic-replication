# -*- coding: utf-8 -*-
""" Result.Result class to wrap the simulation results."""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "SlyJabiru"
__version__ = "0.1"

import numpy as np


class StrategicResult(object):
    """ Result accumulators."""

    # , delta_t_save=1):
    def __init__(self, nbArms, horizon, nbArmsPerAgents, means, bestArmMean,
                 indexes_bestarm=-1):
        """ Create ResultMultiPlayers."""
        # self._means = means  # Keep the means for ChangingAtEachRepMAB cases
        # self.delta_t_save = delta_t_save  #: Sample rate for saving.
        self.means = means
        self.bestArmMean = bestArmMean

        self.choices = np.zeros(horizon, dtype=int)  #: Store all the choices.
        self.rewards = np.zeros(horizon)  #: Store all the rewards, to compute the mean.
        self.pulls = np.zeros(nbArms, dtype=int)  #: Store the pulls.
        self.instantRegrets = np.zeros(horizon)
        
        self.nbArmsPerAgents = nbArmsPerAgents
        self.agentChoices = np.zeros(horizon, dtype=int)
        self.agentChosenNb = np.zeros(len(nbArmsPerAgents), dtype=int)

        self.rewardsPerArms = np.zeros(nbArms)
        self.rewardsPerAgents = np.zeros(len(nbArmsPerAgents))
        
        # if means is not None:
        #     indexes_bestarm = np.nonzero(np.isclose(means, np.max(means)))[0]
        # indexes_bestarm = np.asarray(indexes_bestarm)
        # if np.size(indexes_bestarm) == 1:
        #     indexes_bestarm = np.asarray([indexes_bestarm])
        # self.indexes_bestarm = [ indexes_bestarm for _ in range(horizon)]  #: Store also the position of the best arm, XXX in case of dynamically switching environment.
        # self.running_time = -1  #: Store the running time of the experiment.
        # self.memory_consumption = -1  #: Store the memory consumption of the experiment.
        # self.number_of_cp_detections = 0  #: Store the number of change point detected during the experiment.

    def store(self, time, choice, reward):
        """ Store results."""
        self.choices[time] = choice  # 몇 번 arm 을 뽑았는가?
        self.rewards[time] = reward
        self.pulls[choice] += 1  # 각 arm 을 몇 번 뽑았는가?
        self.instantRegrets[time] = self.bestArmMean - self.means[choice]
        
        armPossession = np.cumsum(self.nbArmsPerAgents) - 1
        temp = (armPossession >= choice)
        agent = np.where(temp)[0][0]
        
        self.agentChoices[time] = agent
        self.agentChosenNb[agent] += 1

        # 여기서, agent 당 reward 를 만들어서 올려주자.
        # 그리고, strategic evaluator 에서 받아주면 됨!
        self.rewardsPerArms[choice] += reward
        self.rewardsPerAgents[agent] += reward


    # def change_in_arms(self, time, indexes_bestarm):
    #     """ Store the position of the best arm from this list of arm.

    #     - From that time t **and after**, the index of the best arm is stored as ``indexes_bestarm``.

    #     .. warning:: FIXME This is still experimental!
    #     """
    #     for t in range(time, len(self.indexes_bestarm)):
    #         self.indexes_bestarm[t] = indexes_bestarm

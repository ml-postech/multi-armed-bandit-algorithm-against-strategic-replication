"""
author: Julien SEZNEC
Plot utility to reproduce Figure 2 of [Seznec et al.,  2019a]
Reference: [Seznec et al.,  2019a]
Rotting bandits are not harder than stochastic ones;
Julien Seznec, Andrea Locatelli, Alexandra Carpentier, Alessandro Lazaric, Michal Valko ;
Proceedings of Machine Learning Research, PMLR 89:2564-2572, 2019.
http://proceedings.mlr.press/v89/seznec19a.html
https://arxiv.org/abs/1811.11043 (updated version)
"""
from SMPyBandits.Policies import wSWA, FEWA, EFF_FEWA, RAWUCB, EFF_RAWUCB, EFF_RAWUCB_pp, GaussianGLR_IndexPolicy, \
  klUCBloglog_forGLR, DiscountedUCB as DUCB, SWUCB, Exp3S
import os
import numpy as np
from numpy import format_float_scientific
from matplotlib import pyplot as plt

plt.style.use('seaborn-colorblind')
plt.style.use('style.mplstyle')


def fig2(data, name='fig2.pdf', ylim=2400, ylim2=500, freq=50, leg=0.2):
  # --------------  PLOT  --------------
  legend_size = leg
  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, legend_size]}, figsize=(12, 10))
  N_arms = 9
  ind = np.arange(N_arms)  # the x locations for the groups
  width = 0.7  # the width of the bars
  L = np.array([0.001 * np.sqrt(10) ** (i) for i in range(9)])
  freq = freq
  for i, policy in enumerate(data):
    T = data[policy]["mean"].shape[0]
    X = range(T)
    ax1.plot(X[::freq], data[policy]["mean"][::freq], linewidth=3, color=None if i != 6 else 'gray')
    color = ax1.get_lines()[-1].get_c()
    ax1.plot(X[::freq], data[policy]["uppq"][::freq], linestyle='--', color=color, linewidth=1)
    ax1.plot(X[::freq], data[policy]["lowq"][::freq], linestyle='--', color=color, linewidth=1)
    ax1.fill_between(X[::freq], data[policy]["uppq"][::freq], data[policy]["lowq"][::freq], alpha=.05, color=color)
    height = data[policy]["pull"][1:] * L
    x_pos = ind - width / 2 + (i + 2) * width / len(data)
    width_bar = width / len(data)
    ax2.bar(x_pos, height, width_bar, bottom=0, label=policy, color=color)
    for j in np.argwhere(height > ylim2):
      ax2.text(x_pos[j], ylim2 * 1.01, int(height[j]), ha='center', va='bottom', rotation='vertical',
               fontsize=18, color=color)
  ax1.set_ylim(0, ylim)
  ax1.set_xlim(0, T)
  ax1.set_xlabel('Round ($t$)')
  ax1.set_ylabel('Average regret $R_t$')
  ax1.xaxis.set_label_coords(0.5, -0.08)
  ax1.grid(False)
  ax2.set_xticks(ind + width / len(data))
  xticks = [format_float_scientific(mu, exp_digits=1, precision=0) for mu in L]
  xticks = [float(xtick) if j % 2 == 0 else '' for j, xtick in enumerate(xticks)]
  ax2.set_ylim(0, ylim2)
  ax2.set_xticklabels(xticks)
  ax2.set_ylabel('Average regret per arm $R_T^i$ ($T = 25000$)')
  ax2.set_xlabel("Arm's $\Delta_i$")
  ax2.grid(False)
  ax2.xaxis.set_label_coords(0.5, -0.08)
  ax2.yaxis.set_label_coords(-0.08, 0.5)
  handles, labels = ax2.get_legend_handles_labels()
  pos = ax3.get_position()
  fig.legend(handles, labels, loc=[0.9 * pos.x0 + 0.1 * pos.x1, (pos.y1 - pos.y0) / 2], prop={'variant': 'small-caps'},
             edgecolor='k')
  ax3.grid(False)
  ax3.axis('off')
  # Hide axes ticks
  ax3.set_xticks([])
  ax3.set_yticks([])
  # -------------- SAVE --------------
  fig.set_size_inches(30, 10)
  fig.tight_layout()
  fig.savefig(name)


def preproc_plot_fig2(policies, name='fig2.pdf', ylim=2400, ylim2=500, freq=50, leg=0.2):
  data = {}
  for policy in policies:
    policy_name = str(policy[0](nbArms=2, **policy[1]))
    policy_name_nospace = policy_name.replace(' ', '_')
    policy_data_regret = [
      np.load(os.path.join('./data', file))
      for file in os.listdir('./data') if
      file.startswith("REGRET_" + policy_name_nospace)
    ]
    policy_data_pull = [
      np.load(os.path.join('./data', file))
      for file in os.listdir('./data') if
      file.startswith("DIFFPULL_" + policy_name_nospace)
    ]
    if not policy_data_regret:
      print(policy_name_nospace)
      continue
    regret_data_array = np.concatenate(policy_data_regret, axis=0)
    pull_data_array = np.concatenate(policy_data_pull, axis=0)
    data[policy_name] = {
      "mean": regret_data_array.mean(axis=0),
      "uppq": np.quantile(regret_data_array, 0.9, axis=0),
      "lowq": np.quantile(regret_data_array, 0.1, axis=0),
      "pull": pull_data_array.mean(axis=0)
    }
  return fig2(data, name=name, ylim=ylim, ylim2=ylim2, leg=leg)


# if __name__ == "__main__":
  # policies = [
  #   [RAWUCB, {'alpha': 1.4}],
  #   [RAWUCB, {'alpha': 4}],
  #   [FEWA, {'alpha': .06}],
  #   [FEWA, {'alpha': 4}],
  #   [wSWA, {'alpha': 0.002}],
  #   [wSWA, {'alpha': 0.02}],
  #   [GaussianGLR_IndexPolicy, {'policy': klUCBloglog_forGLR, 'delta': np.sqrt(1 / 25000), 'alpha0': 0,
  #                              'per_arm_restart': True, 'sig2': 1, 'use_localization': False}],
  # ]
  # preproc_plot_fig2(policies, name="fig2_main.pdf", ylim=1400, ylim2=700)
  #
  # policies = [
  #   [wSWA, {'alpha': 0.002}],
  #   [wSWA, {'alpha': 0.02}],
  #   [wSWA, {'alpha': 0.2}],
  #   [DUCB, {'gamma': 0.997}],
  #   [SWUCB, {'tau': 200}],
  #   [GaussianGLR_IndexPolicy, {'policy': klUCBloglog_forGLR, 'delta': np.sqrt(1 / 25000), 'alpha0': 0,
  #                              'per_arm_restart': True, 'sig2': 1, 'use_localization': False}],
  #   [Exp3S, {}],
  #   # [EFF_RAWUCB_pp2, {'alpha': 1.4, 'm': 1.01}],  # 12
  # ]
  # preproc_plot_fig2(policies, name="fig2_SWA.pdf", ylim=5000, ylim2=750)

  # policies = [
  #   [RAWUCB, {'alpha': 1.4}],  # 7
  #   [EFF_RAWUCB, {'alpha': 1.4, 'm': 1.1}],  # 12
  #   [EFF_RAWUCB, {'alpha': 1.4, 'm': 2}],  # 13
  # ]
  # preproc_plot_fig2(policies, name="fig2_eff.pdf", ylim=400, ylim2=150, leg = 0.4)

  # policies = [
  #   [RAWUCB, {'alpha': 1.4}],  # 7
  #   [EFF_RAWUCB_pp, {'alpha': 1.4, 'm': 1.01}],  # 14
  # ]
  # preproc_plot_fig2(policies, name="fig2_pp.pdf", ylim=1400, ylim2=700)

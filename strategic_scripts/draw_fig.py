import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from cycler import cycler
import numpy as np
import json
import ujson
from os import listdir
from os.path import isfile, join
from collections import Counter


plt.style.use("default")

plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.spines.left'] = True
plt.rcParams['font.size'] = 20 # For axvline label size
plt.rcParams['figure.dpi'] = 600
plt.rcParams["figure.figsize"] = (8, 8)

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.grid'] = False

color_cycle = ['#2ca02c',
               '#1f77b4',
               '#ff7f0e',
               '#d62728',
               # '#14c7d9'
               # '#f582dc'
              ]
marker_cycle = ['o', 's', '^', 'D']
marker_facecolor_cycle = ['None', 'None', '#ff7f0e', '#d62728']
marker_edgecolor_cycle = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
marker_size_cycle = [11, 11, 15, 11]
marker_edgewidth_cycle = [4.5, 4.5, 1, 1]
linestyle_cycle = [
    (0, (1, 0)),
    (0, (4.5, 1.5)),
    (0, (1, 1)),
    (0, (2.5, 1, 1, 1))
]

def check_data(remove_list, result_dir='results'):
    onlyfiles = sorted([f for f in listdir(result_dir) if isfile(join(result_dir, f))])
    check_list = []

    for fname in onlyfiles:
        temp = fname
        for remove in remove_list:
            idx = temp.find(remove)
            if idx != -1:
                temp = temp[0:idx]
        check_list.append(temp)

    counter = dict(Counter(check_list))
    # print(counter)
    for it in counter.values():
        assert it == len(remove_list)


def make_agent_reward_dict(
    fig_experiment_list,
    agent_index,
    fig_L,
    result_dir='results/', repetition=100
):
    fig_rewards_mean = {
        "UCB": [],
        "H_UCB": [],
        "RH_UCB": [],
    }

    fig_rewards_std = {
        "UCB": [],
        "H_UCB": [],
        "RH_UCB": [],
    }

    for l in fig_L:
        fig_rewards_mean[f"Sampled_R_UCB_{l}"] = []
        fig_rewards_std[f"Sampled_R_UCB_{l}"] = []

    for expr in fig_experiment_list:
        policies = list(fig_rewards_mean.keys())
        for policy in policies:
            fname = join(result_dir, f'{expr}_{policy}.json')
            pol_str = policy
            for l in fig_L:
                pol_str = pol_str.replace(f'_{l}', '')
            with open(fname, 'r') as f:
                data = ujson.load(f)
                assert repetition == data['env_0']['Repetitions']
                
                result = data['env_0']['Result'][pol_str]
                agent_reward = result['rewardsPerAgentsMean'][agent_index]
                agent_std = result['rewardsPerAgentsStd'][agent_index]
                # print(f'{pol_str}: {agent_std}')
                fig_rewards_mean[policy].append(agent_reward)
                fig_rewards_std[policy].append(agent_std)
    # print()
    return fig_rewards_mean, fig_rewards_std


def make_regret_dict(
    fig_experiment, fig_L,
    result_dir='results/', repetition=100
):
    fig_regret_mean = {
        "UCB": [],
        "H_UCB": [],
        "RH_UCB": [],
    }

    fig_regret_std = {
        "UCB": [],
        "H_UCB": [],
        "RH_UCB": [],
    }

    for l in fig_L:
        fig_regret_mean[f"Sampled_R_UCB_{l}"] = []
        fig_regret_std[f"Sampled_R_UCB_{l}"] = []

    policies = list(fig_regret_mean.keys())
    for policy in policies:
        fname = f"results/{fig_experiment}_{policy}.json"
        pol_str = policy
        for l in fig_L:
            pol_str = pol_str.replace(f'_{l}', '')
        with open(fname, 'r') as f:
            data = ujson.load(f)
            assert repetition == data['env_0']['Repetitions']

            result = data['env_0']['Result'][pol_str]
            regret_mean = result['cumulatedRegretMean']
            regret_std = result['cumulatedRegretStd']
            # print(f'{pol_str}: {regret_std[-1]}')
            fig_regret_mean[policy] = regret_mean
            fig_regret_std[policy] = regret_std
    # print()
    return fig_regret_mean, fig_regret_std


def plot_fig_a(
    fig_experiment_list, agent_index,
    fig_L=['L5'], result_dir='results/',
    repetition=100,
    figsize=(8,6), dpi=600,
    linewidth=3,
    xlabels=[1, 100, 200, 300, 400, 500],
    yticks=[], yticklabels=[],
    tick_fontsize=20, legend_fontsize=20,
    label_fontsize=25, title_fontsize=30,
    markersize=9,  markeredgewidth=6,
    linestyle='dashed', ci_alpha=0.3,
    pad_inches = 0.1,
    save_pdf=True, save_png=False,
    set_up_str=None, show_setup=False,
    plot_show=False
):
    
    fig_rewards_mean, fig_rewards_std = make_agent_reward_dict(fig_experiment_list,
                                                               agent_index,
                                                               fig_L)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor('None')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    x = range(len(xlabels))
    plt.xticks(ticks=x,
               labels=xlabels,
               fontsize=tick_fontsize,
              )
    plt.yticks(fontsize=tick_fontsize)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10000))
    
    plot_policies = list(fig_rewards_mean.keys())
    
    for idx, policy in enumerate(plot_policies):
        pol_str = policy
        if 'Sampled_R_UCB' in pol_str:
            l_idx = pol_str.find('L')
            pol_str = f'S-UCB({pol_str[l_idx+1:]})'
        pol_str = pol_str.replace('_', '-')
        if pol_str == 'UCB':
            pol_str = 'UCB1'
            
        interval_size = np.array(fig_rewards_std[policy])
        lower_bound = fig_rewards_mean[policy] - interval_size
        upper_bound = fig_rewards_mean[policy] + interval_size
        plt.plot(
            fig_rewards_mean[policy], label=pol_str,
            color=color_cycle[idx], linestyle=linestyle, linewidth=linewidth,
            marker=marker_cycle[idx], markersize=marker_size_cycle[idx], markeredgewidth=marker_edgewidth_cycle[idx],
            markerfacecolor=marker_facecolor_cycle[idx], markeredgecolor=marker_edgecolor_cycle[idx]
        )
        ax.fill_between(x, lower_bound, upper_bound, color=color_cycle[idx], alpha=ci_alpha)

    if show_setup:
        plt.text(3.3, 4800, set_up_str, fontsize=label_fontsize)

    plt.legend(fontsize=legend_fontsize)
    plt.title('')
    plt.xlabel('N', fontsize=label_fontsize)
    plt.ylabel('Reward of agent 0.5', fontsize=label_fontsize)
    plt.yticks(ticks=[0, 10000, 20000, 30000, 40000, 50000],
               labels=['0', '10k', '20k', '30k', '40k', '50k'],
               # rotation=45,
               fontsize=tick_fontsize
              )
    plt.tight_layout()
    
    if show_setup:
        fig.text(0.25, 0.24, set_up_str, fontsize=label_fontsize)
    fig.tight_layout()
    fig.set_size_inches(figsize)
    # fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

    if save_pdf:
        fig.savefig(f'./figs/fig_a.pdf',
                    format='pdf', dpi=dpi,
                    bbox_inches='tight',pad_inches=pad_inches)
    if save_png:
        fig.savefig(f'./figs/fig_a.png',
                    format='png', dpi=dpi,
                    bbox_inches='tight',pad_inches=pad_inches)
    if plot_show:
        plt.show()

    # is_transparent = True
    # fig.savefig(f'./figs/fig-(a).png',
    #             # facecolor='None',
    #             # facecolor='white' if face_color else 'None',
    #             # transparent=is_transparent,
    #             format='png', dpi=dpi)


def plot_fig_b(
    fig_experiment_list, agent_index,
    fig_L=['L5'], result_dir='results/',
    repetition=100,
    figsize=(8,6),dpi=600,
    linewidth=3,
    xlabels=[1, 100, 200, 300, 400, 500],
    yticks=[], yticklabels=[],
    tick_fontsize=20, legend_fontsize=20,
    label_fontsize=25, title_fontsize=30,
    markersize=9,  markeredgewidth=6,
    linestyle='dashed', ci_alpha=0.3,
    pad_inches = 0.1,
    save_pdf=True, save_png=False,
    set_up_str=None, show_setup=False,
    plot_show=False
):

    fig_rewards_mean, fig_rewards_std = make_agent_reward_dict(fig_experiment_list,
                                                               agent_index,
                                                               fig_L)

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]},
                                   figsize=figsize, dpi=dpi)
    ax1.set_facecolor('None')
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(True)
    ax2.set_facecolor('None')
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['left'].set_visible(True)

    x = range(len(xlabels))
    plt.setp([ax1, ax2], xticks=x, xticklabels=xlabels)
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    plot_policies = list(fig_rewards_mean.keys())
    for idx, policy in enumerate(plot_policies):
        pol_str = policy
        if 'Sampled_R_UCB' in pol_str:
            l_idx = pol_str.find('L')
            pol_str = f'S-UCB({pol_str[l_idx+1:]})'
        pol_str = pol_str.replace('_', '-')
        if pol_str == 'UCB':
            pol_str = 'UCB1'

        interval_size = np.array(fig_rewards_std[policy])
        lower_bound = fig_rewards_mean[policy] - interval_size
        upper_bound = fig_rewards_mean[policy] + interval_size
                
        ax1.plot(
            fig_rewards_mean[policy], label=pol_str,
            color=color_cycle[idx], linestyle=linestyle, linewidth=linewidth,
            marker=marker_cycle[idx], markersize=marker_size_cycle[idx], markeredgewidth=marker_edgewidth_cycle[idx],
            markerfacecolor=marker_facecolor_cycle[idx], markeredgecolor=marker_edgecolor_cycle[idx]
        )
        ax1.fill_between(x, lower_bound, upper_bound, color=color_cycle[idx], alpha=ci_alpha)
        
        ax2.plot(
            fig_rewards_mean[policy], label=pol_str,
            color=color_cycle[idx], linestyle=linestyle, linewidth=linewidth,
            marker=marker_cycle[idx], markersize=marker_size_cycle[idx], markeredgewidth=marker_edgewidth_cycle[idx],
            markerfacecolor=marker_facecolor_cycle[idx], markeredgecolor=marker_edgecolor_cycle[idx]
        )
        ax2.fill_between(x, lower_bound, upper_bound, color=color_cycle[idx], alpha=ci_alpha)

    ax1.set_ylim(86500, 90500)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax1.set_yticks([87000, 88000, 89000, 90000])
    ax1.set_yticklabels(['87k', '88k', '89k', '90k'])
    
    ax2.set_ylim(38200, 40200)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax2.set_yticks([38000, 39000])
    ax2.set_yticklabels(['38k', '39k'])

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False, 
        labelbottom=False,
        labeltop=False
    )
    ax2.xaxis.tick_bottom()

    handles, labels = ax1.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)

    # Adds slanted lines to axes
    d = .3  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=20,
        linestyle='none',
        color='k',
        mec='k',
        mew=2.0,
        clip_on=False
    )
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    fig.tight_layout()
    if show_setup:
        fig.text(0.25, 0.24, set_up_str, fontsize=label_fontsize)

    ax1.set_title('')
    ax2.set_xlabel('N', fontsize=label_fontsize)
    ax1.set_ylabel('Reward of agent 0.9', fontsize=label_fontsize)
    ax1.yaxis.set_label_coords(-0.11, 0.15)

    fig.legend(handle_list, label_list,
               fontsize=legend_fontsize,
               bbox_to_anchor=(0.90, 0.61))
    plt.tight_layout()
    fig.set_size_inches(figsize)

    if save_pdf:
        fig.savefig(f'./figs/fig_b.pdf',
                    format='pdf', dpi=dpi,
                    bbox_inches='tight',pad_inches=pad_inches)
    if save_png:
        fig.savefig(f'./figs/fig_b.png',
                    format='png', dpi=dpi,
                    bbox_inches='tight',pad_inches=pad_inches)
    if plot_show:
        plt.show()


def plot_regret(
    fig_experiment, fig_alphabet, horizon=100000,
    fig_L=['L15'], result_dir='results/',
    repetition=100,
    figsize=(8,6), dpi=600,
    linewidth=3,
    xlabels=[1, 100, 200, 300, 400, 500],
    yticks=[],
    yticklabels=[],
    tick_fontsize=20, legend_fontsize=20,
    label_fontsize=25, title_fontsize=30,
    linestyle='dashed', ci_alpha=0.3,
    pad_inches = 0.1,
    save_pdf=True, save_png=False,
    set_up_str=None, show_setup=False,
    plot_show=False
):
    
    fig_regret_mean, fig_regret_std = make_regret_dict(fig_experiment, fig_L)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor('None')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    if fig_alphabet == 'd':
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5000))
    else:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10000))

    plot_policies = list(fig_regret_mean.keys())
    
    for idx, policy in enumerate(plot_policies):
        pol_str = policy
        if 'Sampled_R_UCB' in pol_str:
            l_idx = pol_str.find('L')
            pol_str = f'S-UCB({pol_str[l_idx+1:]})'
        pol_str = pol_str.replace('_', '-')
        if pol_str == 'UCB':
            pol_str = 'UCB1'

        interval_size = np.array(fig_regret_std[policy])
        lower_bound = fig_regret_mean[policy] - interval_size
        upper_bound = fig_regret_mean[policy] + interval_size
        ax.plot(fig_regret_mean[policy], label=pol_str, color=color_cycle[idx],
                linewidth=linewidth, linestyle=linestyle_cycle[idx])
        ax.fill_between(range(horizon), lower_bound, upper_bound,
                        color=color_cycle[idx], alpha=ci_alpha)

    if show_setup:
        plt.text(3.3, 4800, set_up_str, fontsize=label_fontsize)

    plt.legend(fontsize=legend_fontsize)
    plt.title('')
    plt.xlabel('T', fontsize=label_fontsize)
    plt.ylabel('Cumulative regret', fontsize=label_fontsize)
    plt.xticks(
        ticks=[0, 20000, 40000, 60000, 80000, 100000],
        labels=['$0$', '$2$', '$4$', '$6$', '$8$', '$10$'],
               # rotation=45,
        fontsize=tick_fontsize
    )
    plt.yticks(ticks=yticks,
               labels=yticklabels,
               # rotation=45,
               fontsize=tick_fontsize
              )
    plt.tight_layout()
    if show_setup:
        fig.text(0.25, 0.24, set_up_str, fontsize=label_fontsize)
        
    fig.text(0.85, 0.065, r'$ \times 10^{4}$', fontsize=tick_fontsize-2)
    plt.tight_layout()
    fig.set_size_inches(figsize)
    
    if save_pdf:
        fig.savefig(f'./figs/fig_{fig_alphabet}.pdf',
                    format='pdf', dpi=dpi,
                    bbox_inches='tight',pad_inches=pad_inches)
    if save_png:
        fig.savefig(f'./figs/fig_{fig_alphabet}.png',
                    format='png', dpi=dpi,
                    bbox_inches='tight',pad_inches=pad_inches)
    if plot_show:
        plt.show()
    

if __name__=='__main__':
    check_data(remove_list=['_Sampled_R_UCB_L15', '_Sampled_R_UCB_L5', '_RH_UCB', '_H_UCB', '_UCB'])
    
    fig_a_expr_list = [
        "N100_default",
        "N100_05X100",
        "N100_05X200",
        "N100_05X300",
        "N100_05X400",
        "N100_05X500",
    ]
    fig_a_agent_index = 4
    fig_a_set_up_str = """
    setup = {
        0: [0.9] X 1
        1: [0.8] X 1
        2: [0.7] X 1
        3: [0.6] X 1
        4: [0.5] X N
    }
    """
    plot_fig_a(
        fig_experiment_list=fig_a_expr_list,
        agent_index=fig_a_agent_index,
        set_up_str=fig_a_set_up_str,
        save_pdf=True, save_png=True,
        linewidth=3, ci_alpha=0.2,
        markersize=12, markeredgewidth=5,
        pad_inches = 0.05,
    )
    with open('figs/fig_a.txt', 'w') as f:
        f.write(fig_a_set_up_str)
    
    fig_b_expr_list = [
        "N100_default",
        "N100_09X100",
        "N100_09X200",
        "N100_09X300",
        "N100_09X400",
        "N100_09X500",
    ]
    fig_b_agent_index = 0
    fig_b_set_up_str = """
    setup = {
        0: [0.9] X N
        1: [0.8] X 1
        2: [0.7] X 1
        3: [0.6] X 1
        4: [0.5] X 1
    }
    """
    plot_fig_b(
        fig_experiment_list=fig_b_expr_list,
        agent_index=fig_b_agent_index,
        set_up_str=fig_b_set_up_str,
        save_pdf=True, save_png=True,
        linewidth=3, ci_alpha=0.2,
        markersize=12, markeredgewidth=5,
        pad_inches = 0.05,
    )
    with open('figs/fig_b.txt', 'w') as f:
        f.write(fig_b_set_up_str)
    
    fig_c_experiment = 'N100_single_origin_arm1000X1'
    fig_c_alphabet = 'c'
    fig_c_set_up_str = """
    setup = {
        0: [0.9] X 1
        1: [0.8] X 1
        2: [0.7] X 1
        3: [0.6] X 1
        4: [0.5] X 1000
    }
    """
    plot_regret(
        fig_experiment=fig_c_experiment,
        fig_alphabet=fig_c_alphabet,
        fig_L=['L5'],
        save_pdf=True, save_png=True,
        linewidth=5, ci_alpha=0.2,
        yticks=[10000, 20000, 30000, 40000],
        yticklabels=['10k', '20k', '30k', '40k'],
        pad_inches = 0.05,
    )
    with open('figs/fig_c.txt', 'w') as f:
        f.write(fig_c_set_up_str)
    
    fig_d_experiment = 'N100_single_origin_arm1000X4'
    fig_d_alphabet = 'd'
    fig_d_set_up_str = """
    setup = {
        0: [0.9] X 1
        1: [0.8] X 1000
        2: [0.7] X 1000
        3: [0.6] X 1000
        4: [0.5] X 1000
    }
    """
    plot_regret(
        fig_experiment=fig_d_experiment,
        fig_alphabet=fig_d_alphabet,
        fig_L=['L5'],
        save_pdf=True, save_png=True,
        linewidth=5, ci_alpha=0.2,
        yticks=[5000, 10000, 15000, 20000],
        yticklabels=['5k', '10k', '15k', '20k'],
        pad_inches = 0.05,
    )
    with open('figs/fig_d.txt', 'w') as f:
        f.write(fig_d_set_up_str)
    
    fig_e_experiment = 'N100_rh_ucb_best_10_100_replicate1000X3'
    fig_e_alphabet = 'e'
    fig_e_set_up_str = """
    setup = {
        0: [0.9, 0.2, 0.1] X [10, 100, 100]
        1: [0.8, 0.2, 0.1] X [10, 100, 100]
        2: [0.7, 0.2, 0.1] X [1000, 1000, 1000]
        3: [0.6, 0.2, 0.1] X [1000, 1000, 1000]
        4: [0.5, 0.2, 0.1] X [1000, 1000, 1000]
    }
    """
    plot_regret(
        fig_experiment=fig_e_experiment,
        fig_alphabet=fig_e_alphabet,
        fig_L=['L15'],
        save_pdf=True, save_png=True,
        linewidth=5, ci_alpha=0.2,
        yticks=[10000, 20000, 30000, 40000, 50000],
        yticklabels=['10k', '20k', '30k', '40k', '50k'],
        pad_inches = 0.05,
    )
    with open('figs/fig_e.txt', 'w') as f:
        f.write(fig_e_set_up_str)

# pdf2ps -dLanguageLevel=3 fig_a.pdf fig_a.eps

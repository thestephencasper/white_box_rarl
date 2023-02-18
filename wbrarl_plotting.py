from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import stats

rc('font', **{'family': 'serif', 'serif': ['Palatino']})
plt.rcParams['pdf.fonttype'] = 42

results_path = Path('./results/')
N_TRAIN_STEPS = 1000000
FS = 15
N_EXCLUDE = 20
TOTAL_N = 40

#this needs to be copied over from wbrarl.py
COEF_DICT = {'HalfCheetah-v3': {'mass': [0.2, 0.3, 0.4, 0.5, 1.5, 2.0, 2.5, 3.0],
                                'friction': [0.05, 0.1, 0.2, 0.3, 1.3, 1.5, 1.7, 1.9]},
             'Hopper-v3': {'mass': [0.2, 0.3, 0.4, 0.5, 1.05, 1.1, 1.15, 1.2],
                           'friction': [0.2, 0.3, 0.4, 0.5, 1.4, 1.6, 1.8, 2.0]},
             'Simglucose': {'kp1': [0.4, 0.5, 0.6, 0.7, 1.3, 1.4, 1.5, 1.6],
                            'ka1': [0.4, 0.5, 0.6, 0.7, 1.3, 1.4, 1.5, 1.6]},
             'Cancer': {'gamma': [0.3, 0.4, 0.5, 0.6, 1.66, 2, 2.5, 3.33],
                        'lambda_p': [0.3, 0.4, 0.5, 0.6, 1.66, 2, 2.5, 3.33]},
             }


def get_env_name(names):
    if 'Cheetah' in names[0]:
        env_name = 'HalfCheetah-v3'
    elif 'Hopper' in names[0]:
        env_name = 'Hopper-v3'
    elif 'Simglucose' in names[0]:
        env_name = 'Simglucose'
    elif 'Cancer' in names[0]:
        env_name = 'Cancer'
    else:
        raise NotImplementedError
    return env_name


def get_fs(names):
    if 'Cheetah' in names[0]:
        f1 = 'mass'
        f2 = 'friction'
    elif 'Hopper' in names[0]:
        f1 = 'mass'
        f2 = 'friction'
    elif 'Simglucose' in names[0]:
        f1 = 'kp1'
        f2 = 'ka1'
    elif 'Cancer' in names[0]:
        f1 = 'gamma'
        f2 = 'lambda_p'
    else:
        raise NotImplementedError
    return f1, f2


def get_learning_curves(fnames):

    paths = results_path.glob(fnames + '*')
    ys = []
    for path in paths:
        with open(str(path), 'rb') as f:
            y = pickle.load(f)
            ys.append(y)

    ys = np.array(ys)
    idxs = np.argsort(ys[:, -1])
    ys = ys[idxs[N_EXCLUDE:]]

    return np.array(ys)


def heatmap(data, row_labels, col_labels, ax=None, col_i=0, do_cbar=True,
            cbar_kw={}, cbarlabel="", **kwargs):

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if do_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(col_labels)
    if col_i == 0:
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_yticklabels(labels=row_labels)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None,
                     textcolors=("black", "white"),
                     threshold=None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    # if isinstance(valfmt, str):
    #     valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, int(data[i, j]), **kw)
            texts.append(text)

    return texts


def plot_and_print_eval(eval_names, do_cbar=True):

    eval_results = []
    all_hm_means = []
    for names in eval_names:
        paths = results_path.glob(names + '_eval*')
        hmaps = []
        for path in paths:
            with open(str(path), 'rb') as f:
                data = pickle.load(f)
                hmaps.append(np.array(data))
        hmaps = np.array(hmaps)
        hm_means = np.mean(hmaps, axis=(1, 2))
        idxs = np.argsort(hm_means)
        hmaps = hmaps[idxs[N_EXCLUDE:]]
        if len(hmaps) > 0:
            hmap_means = np.mean(hmaps, axis=0)
            hmap_means = np.round(hmap_means)
            eval_results.append(hmap_means)
            all_hm_means.append([np.mean(hm) for hm in hmaps])

    env_name = get_env_name(eval_names)
    f1, f2 = get_fs(eval_names)

    mass_vals = COEF_DICT[env_name][f1]
    friction_vals = COEF_DICT[env_name][f2]
    xvals = [str(v) + f'x {f1}' for v in mass_vals]
    yvals = [str(v) + f'x {f2}' for v in friction_vals]
    titles = ['RL', 'RARL', 'Latent/Action WB-RARL']
    all_min = min([np.min(er) for er in eval_results])
    all_max = max([np.max(er) for er in eval_results])

    wid = 18 if do_cbar else 16
    fig, axes = plt.subplots(1, 3, figsize=(wid, 6))
    for i in range(len(eval_results)):
        im, cbar = heatmap(eval_results[i], xvals, yvals, ax=axes[i], col_i=i, do_cbar=do_cbar,
                           cmap="YlGn", **{'vmin': all_min, 'vmax': all_max})
        _ = annotate_heatmap(im)
        fig.tight_layout()
        axes[i].set_title(titles[i] + f', n={TOTAL_N - N_EXCLUDE}', fontsize=FS+4)

    axes[0].set_ylabel(env_name.replace('-v3', ''), fontsize=FS+8)
    fig.tight_layout(rect=[0, 0.15, 0.95, 1.0])

    all_hm_means = np.array(all_hm_means)
    all_means = []
    for hm in all_hm_means:
        print(np.mean(np.array(hm)), np.std(np.array(hm)))
        all_means.append(int(np.round(np.mean(np.array(hm)))))

    pairs = [(0, 1), (0, 2), (1, 2)]
    all_ps = []
    for pair in pairs:
        result = stats.ttest_ind(all_hm_means[pair[0]], all_hm_means[pair[1]], alternative='less')
        print(pair, result)
        all_ps.append(round(result[1], 3))

    axes[0].set_xlabel(f'Mean: {all_means[0]}\nWB-RARL v. RL p: {all_ps[1]}', fontsize=FS+3)
    axes[1].set_xlabel(f'Mean: {all_means[1]}\nRARL v. RL p: {all_ps[2]}', fontsize=FS+3)
    axes[2].set_xlabel(f'Mean: {all_means[2]}', fontsize=FS+3)

    plt.show()

    # plt.imsave(f"{env_name.replace('-v3', '')}_eval.pdf", fig)
    fig.savefig(f"./{env_name.replace('-v3', '')}_eval_cbar={do_cbar}.pdf", format='pdf', bbox_inches='tight')


def plot_learning_curves_and_row_col_means(results_names, eval_names, errbar=True):

    env_name = get_env_name(results_names)
    f1, f2 = get_fs(eval_names)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['royalblue', 'firebrick', 'darkorange']
    labels = ['RL', 'RARL', 'Latent/Action WB-RARL']

    # plot the learning curves
    ys = []
    y_means = []
    for name in results_names:
        ys.append(get_learning_curves(name))
    for i, y in enumerate(ys):
        x = np.linspace(1, N_TRAIN_STEPS, num=len(y[0]))
        y_mean = np.mean(y, axis=0)
        axes[0].plot(x, y_mean, color=colors[i])
    axes[0].legend(labels[:len(results_names)], fontsize=FS, loc='lower right')
    for i, y in enumerate(ys):
        x = np.linspace(1, N_TRAIN_STEPS, num=len(y[0]))
        y_mean = np.mean(y, axis=0)
        y_means.append(y_mean)
        y_sem = stats.sem(y, axis=0)
        axes[0].fill_between(x, y_mean-y_sem, y_mean+y_sem, color=colors[i], alpha=0.2)
    axes[0].set_title(env_name.replace('-v3', '') + f' Train, n={TOTAL_N - N_EXCLUDE}', fontsize=FS+2)
    axes[0].set_ylabel('Reward per Episode', fontsize=FS)
    axes[0].set_xlabel('t', fontsize=FS)
    axes[0].grid(alpha=0.4)

    all_rowmeans = []
    all_colmeans = []
    for names in eval_names:
        paths = results_path.glob(names + '_eval*')
        hmaps = []
        for path in paths:
            with open(str(path), 'rb') as f:
                data = pickle.load(f)
                hmaps.append(np.array(data))

        hmaps = np.array(hmaps)
        hm_means = np.mean(hmaps, axis=(1, 2))
        idxs = np.argsort(hm_means)
        hmaps = hmaps[idxs[N_EXCLUDE:]]

        if len(hmaps) > 0:
            all_rowmeans.append(np.mean(hmaps, axis=1))
            all_colmeans.append(np.mean(hmaps, axis=2))

    all_rowmeans = np.array(all_rowmeans)
    all_colmeans = np.array(all_colmeans)

    rowmeans = np.mean(all_rowmeans, axis=1)
    rowsems = stats.sem(all_rowmeans, axis=1)
    colmeans = np.mean(all_colmeans, axis=1)
    colsems = stats.sem(all_colmeans, axis=1)

    mass_vals = COEF_DICT[env_name][f1]
    friction_vals = COEF_DICT[env_name][f2]
    xlbls = [''] + [str(v) + 'x' for v in mass_vals]
    ylbls = [''] + [str(v) + 'x' for v in friction_vals]
    xvals = list(range(rowmeans.shape[1]))

    if errbar:
        for i in range(len(rowmeans)):
            axes[1].errorbar(xvals, colmeans[i], colsems[i], color=colors[i], fmt='s', lw=4, capsize=8, capthick=4,
                             alpha=0.5)
            axes[2].errorbar(xvals, rowmeans[i], rowsems[i], color=colors[i], fmt='s', lw=4, capsize=8, capthick=4,
                             alpha=0.5)

    else:
        for i in range(len(rowmeans)):
            axes[1].plot(xvals, colmeans[i], color=colors[i])
            axes[1].fill_between(xvals, colmeans[i] - colsems[i], colmeans[i] + colsems[i], color=colors[i], alpha=0.2)
            axes[2].plot(xvals, rowmeans[i], color=colors[i])
            axes[2].fill_between(xvals, rowmeans[i] - rowsems[i], rowmeans[i] + rowsems[i], color=colors[i], alpha=0.2)

    axes[1].set_xticklabels(xlbls)
    axes[2].set_xticklabels(ylbls)
    axes[1].set_xlabel(f'{f1.title()} Multiplier', fontsize=FS)
    axes[2].set_xlabel(f'{f2.title()} Multiplier', fontsize=FS)
    axes[1].set_title(env_name.replace('-v3', '') + f' Test, n={TOTAL_N - N_EXCLUDE}', fontsize=FS+2)
    axes[2].set_title(env_name.replace('-v3', '') + f' Test, n={TOTAL_N - N_EXCLUDE}', fontsize=FS+2)
    axes[1].grid(alpha=0.4)
    axes[2].grid(alpha=0.4)

    plt.show()
    fig.savefig(f"./{env_name.replace('-v3', '')}_train_and_eval_errbar={errbar}.pdf", format='pdf', bbox_inches='tight')



hopper_results_names = [
    'agent_control_Hopper-v3_2000000_*_rewards',
    'agent_rarl_Hopper-v3_2000000_id=*_rewards',
    'agent_lat_act_rarl_Hopper-v3_2000000_id=*_rewards']
half_cheetah_results_names = [
    'agent_control_HalfCheetah-v3_2000000_*_rewards',
    'agent_rarl_HalfCheetah-v3_2000000_id=*_rewards',
    'agent_lat_act_rarl_HalfCheetah-v3_2000000_id=*_rewards']
simglucose_results_names = [
    'agent_control_Simglucose_1000000_*_rewards',
    'agent_rarl_Simglucose_1000000_id=*_rewards',
    'agent_lat_act_rarl_Simglucose_1000000_id=*_rewards']
cancer_results_names = [
    'agent_control_Cancer_1000000_*_rewards',
    'agent_rarl_Cancer_1000000_id=*_rewards',
    'agent_lat_act_rarl_Cancer_1000000_id=*_rewards']

half_cheetah_eval_names = ['best_agent_control_HalfCheetah-v3_2000000_id=*',
                           'best_agent_rarl_HalfCheetah-v3_2000000_id=*',
                           'best_agent_lat_act_rarl_HalfCheetah-v3_2000000_id=*']
hopper_eval_names = ['best_agent_control_Hopper-v3_2000000_id=*',
                     'best_agent_rarl_Hopper-v3_2000000_id=*',
                     'best_agent_lat_act_rarl_Hopper-v3_2000000_id=*']
simglucose_eval_names = ['best_agent_control_Simglucose_1000000_id=*',
                     'best_agent_rarl_Simglucose_1000000_id=*',
                     'best_agent_lat_act_rarl_Simglucose_1000000_id=*']
cancer_eval_names = ['best_agent_control_Cancer_1000000_id=*',
                     'best_agent_rarl_Cancer_1000000_id=*',
                     'best_agent_lat_act_rarl_Cancer_1000000_id=*']


# plot_and_print_eval(half_cheetah_eval_names)
# plot_learning_curves_and_row_col_means(half_cheetah_results_names, half_cheetah_eval_names)
#
# plot_and_print_eval(hopper_eval_names)
# plot_learning_curves_and_row_col_means(hopper_results_names, hopper_eval_names)

# plot_and_print_eval(simglucose_eval_names)
# plot_learning_curves_and_row_col_means(simglucose_results_names, simglucose_eval_names)

# plot_and_print_eval(cancer_eval_names)
# plot_learning_curves_and_row_col_means(cancer_results_names, cancer_eval_names)

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import pickle
import xml.etree.ElementTree as ET
from datetime import datetime as dt

from util_functions import read_config


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.

    :param width: float
            Document textwidth or columnwidth in pts
    :param fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    :param subplots: array-like, optional
            The number of rows and columns of subplots.

    :return fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

def parse_date(timestamp, datetime_formats):
    for fmt in datetime_formats:
        try:
            return dt.strptime(timestamp, fmt)
        except ValueError:
            pass
    raise ValueError(f"Time data '{timestamp}' does not match known formats.")

def get_resp_elapsed_time(root, command_name, datetime_formats):
    matching_req = []
    for req in root.findall('./response'):
        for child in req:
            if child.text == command_name:
                matching_req.append(req)

    time_elapsed_list = []
    for req in matching_req:
        start_time_str = req.find('start_time').text
        end_time_str = req.find('end_time').text

        start_time = parse_date(start_time_str, datetime_formats)
        end_time = parse_date(end_time_str, datetime_formats)

        time_elapsed = end_time - start_time
        time_elapsed_list.append(time_elapsed.total_seconds())

    return time_elapsed_list


def compute_backward_transfer(acc_matrix):
    bwt = np.zeros(acc_matrix.shape[0], dtype=float)

    bwt[0] = 0
    for t in range(1, acc_matrix.shape[0]):

        sum = 0
        for i in range(t):
            sum += acc_matrix[t, i] - acc_matrix[i, i]

        bwt[t] = (1 / t) * sum

    return bwt


def plot_class_incr_learning(config, dataset_name, sub_sel_funcs, seq_types, ram_buffer_size, eeprom_buffer_size, num_of_trials):
    eval_metrics = ['acc_test_set_union', 'acc_global', 'bwt']
    class_sequences = np.load(os.path.join(config['artifacts_dir_path'], dataset_name + '_class_sequences.npy'))
    class_seq_len = class_sequences.shape[1]

    fig, ax = plt.subplots(nrows=len(eval_metrics), ncols=len(seq_types), figsize=(8, 10))

    for col, seq_type in enumerate(seq_types):
        for row, eval_metric in enumerate(eval_metrics):
            for func in sub_sel_funcs:

                metric = np.zeros((num_of_trials, class_seq_len - 1), dtype=float)

                for trial in range(num_of_trials):

                    exp_param = f'sub_selection_emulation={str(config["host"]).lower()}_seq={seq_type}_ram_buf_size={ram_buffer_size}_eeprom_buf_size={eeprom_buffer_size}_'
                    if func == 'rand_bal':
                        filename_prefix = f'{dataset_name}_{func}_' + exp_param + f'trial={trial + 1}_'
                        label = func
                    else:
                        num_iter = func[1]
                        filename_prefix = f'{dataset_name}_{func[0]}_' + exp_param + f'num_iter={num_iter}_trial={trial + 1}_'
                        label = f'{func[0]}_{num_iter}_iter'

                    with open(os.path.join(config['results_dir_path'], filename_prefix + 'results_dict.pkl'), 'rb') as f:
                        results_dict = pickle.load(f)

                    if eval_metric != 'bwt':
                        metric[trial, :] = results_dict[eval_metric]
                    else:
                        metric[trial, :] = compute_backward_transfer(results_dict['acc_matrix'])

                metric_mean = np.mean(metric, axis=0)
                metric_std = np.std(metric, axis=0)

                t = np.arange(2, class_seq_len + 1)
                ax[row, col].plot(t, metric_mean, label=label)
                ax[row, col].fill_between(t, metric_mean - metric_std, metric_mean + metric_std, alpha=0.3)
                ax[row, col].grid(True)

                if eval_metric == 'acc_test_set_union':
                    if dataset_name == 'FashionMNIST' or dataset_name == 'MNIST':
                        ax[row, col].set_ylim([0.6, 1.0])
                    else:
                        ax[row, col].set_ylim([0.1, 1.0])
                elif eval_metric == 'acc_global':
                    ax[row, col].set_ylim([0, 1.0])

                if row == 0:
                    ax[row, col].set_title(f'{seq_type} acc seq')

                if col > 0:
                    ax[row, col].set_yticklabels([])



    ax[0, 0].legend()
    ax[0, 0].set_ylabel('ACC test set union')
    ax[1, 0].set_ylabel('ACC global')
    ax[2, 0].set_ylabel('Backward TF')

    ax[2, 0].set_xlabel('Num of classes')
    ax[2, 1].set_xlabel('Num of classes')

    fig.suptitle(f"{dataset_name},\nN_RAM={ram_buffer_size}, N_EEPROM={eeprom_buffer_size}, N_Trials={num_of_trials}")
    plt.savefig(os.path.join(config['plots_dir_path'], f'{dataset_name}_class_incr__ram_buffer_size={ram_buffer_size}_eeprom_buffer_size={eeprom_buffer_size}_n_trials={num_of_trials}.pdf'))
    return fig


def plot_class_incr_learning_2(config, dataset_names, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials, textwidth, color_dict, save_fig=False):
    eval_metrics = ['acc_test_set_union', 'acc_train_set_union']
    dataset_names_str = ''

    width_in, _ = set_size(width=textwidth, subplots=(len(eval_metrics), len(seq_types)*len(dataset_names)))

    fig, ax = plt.subplots(nrows=len(eval_metrics), ncols=len(seq_types)*len(dataset_names), figsize=(width_in, 2.2*2.0))

    for i, dataset_name in enumerate(dataset_names):
        class_sequences = np.load(os.path.join(config['artifacts_dir_path'], dataset_name + '_class_sequences.npy'))
        class_seq_len = class_sequences.shape[1]

        for col, seq_type in enumerate(seq_types):
            for row, eval_metric in enumerate(eval_metrics):
                for func in sub_sel_funcs:
                    for (ram_buffer_size, eeprom_buffer_size) in buffer_sizes:
                        metric = np.zeros((num_of_trials, class_seq_len - 1), dtype=float)

                        for trial in range(num_of_trials):

                            exp_param = f'sub_selection_emulation={str(config["host"]).lower()}_seq={seq_type}_ram_buf_size={ram_buffer_size}_eeprom_buf_size={eeprom_buffer_size}_'
                            if func == 'rand_bal':
                                filename_prefix = f'{dataset_name}_{func}_' + exp_param + f'trial={trial + 1}_'
                            elif func[0] == 'rand_greedy':
                                num_iter = func[1]
                                filename_prefix = f'{dataset_name}_{func[0]}_' + exp_param + f'num_iter={num_iter}_trial={trial + 1}_'
                            elif func[0] == 'evo':
                                num_gen = func[1]
                                filename_prefix = f'{dataset_name}_{func[0]}_' + exp_param + f'num_gen={num_gen}_trial={trial + 1}_'

                            with open(os.path.join(config['results_dir_path'], filename_prefix + 'results_dict.pkl'), 'rb') as f:
                                results_dict = pickle.load(f)

                            if eval_metric != 'bwt':
                                metric[trial, :] = results_dict[eval_metric]
                            else:
                                metric[trial, :] = compute_backward_transfer(results_dict['acc_matrix'])

                        metric_mean = np.mean(metric, axis=0)
                        metric_std = np.std(metric, axis=0)


                        color = 'b'
                        if func == 'rand_bal':
                            linestyle = '-'
                            hatch = '/'
                            label = f'({ram_buffer_size}, {eeprom_buffer_size})'
                        elif func[0] == 'rand_greedy':
                            linestyle = '--'
                            hatch = '//'
                            label = None
                        elif func[0] == 'evo':
                            linestyle = ':'
                            hatch = 'x'
                            label = None

                        t = np.arange(2, class_seq_len + 1)
                        ax[row, 2*i+col].plot(t, metric_mean, linestyle=linestyle, label=label, color=color_dict[(ram_buffer_size, eeprom_buffer_size)])
                        ax[row, 2*i+col].fill_between(t, metric_mean - metric_std, metric_mean + metric_std,
                                                  color=color_dict[(ram_buffer_size, eeprom_buffer_size)], alpha=0.08, linestyle=linestyle, edgecolor='black')
                        ax[row, 2*i+col].grid(True)

                        if eval_metric == 'acc_test_set_union':
                            if dataset_name == 'FashionMNIST' or dataset_name == 'MNIST':
                                ax[row, 2*i+col].set_ylim([0.55, 1.0])
                            else:
                                ax[row, 2*i+col].set_ylim([0.3, 1.0])
                        elif eval_metric == 'acc_train_set_union':
                            if dataset_name == 'FashionMNIST' or dataset_name == 'MNIST':
                                ax[row, 2*i+col].set_ylim([0.55, 1.0])
                            else:
                                ax[row, 2*i+col].set_ylim([0.3, 1.0])
                        elif eval_metric == 'acc_global':
                            ax[row, 2*i+col].set_ylim([0, 1.0])

                        if row == 0:
                            if seq_type == 'low':
                                ax[row, 2*i+col].set_title('$\mathbf{q}_{min}$')
                            elif seq_type == 'high':
                                ax[row, 2*i+col].set_title('$\mathbf{q}_{max}$')

                        if col == 0 and i > 0:
                            ax[row, 2*i+col].set_yticklabels([])
                        if col > 0:
                            ax[row, 2*i+col].set_yticklabels([])

                        if not row == len(eval_metrics) - 1:
                            ax[row, 2*i+col].set_xticklabels([])

                        ax[row, 2*i+col].set_xticks([i for i in range(2, class_seq_len+1, int(class_seq_len / 5))])

        if dataset_name == 'MNIST' or dataset_name == 'EMNIST':
            for j, eval_metric in enumerate(eval_metrics):
                if eval_metric == 'acc_test_set_union':
                    ax[j, 0].set_ylabel('$A_{1}$ on $\mathcal{T}_{t}$')
                elif eval_metric == 'acc_train_set_union':
                    ax[j, 0].set_ylabel('$A_{1}$ on $\{\mathcal{B}_{t}\}_{i=1}^{t}$')

        ax[1, 2*i].set_xlabel('Num of classes')
        ax[1, 2*i+1].set_xlabel('Num of classes')

        dataset_names_str += dataset_name + '_'

    handles, labels = plt.gca().get_legend_handles_labels()

    if len(dataset_names) == 2:
        plt.figtext(0.30, 0.98, dataset_names[0], va="center", ha="center")
        plt.figtext(0.75, 0.98, dataset_names[1], va="center", ha="center")
        fig.legend(title='Volatile and non-volatile mem. buffer sizes (kB):', handles=handles, labels=labels,
                   loc='upper center', bbox_to_anchor=(0.5, 0.03),
                   ncol=4, fancybox=False, shadow=False)
    else:
        plt.figtext(0.53, 0.98, dataset_names[0], va="center", ha="center")
        ax[0, 0].legend(title='Volatile and non-volatile mem.\nbuffer sizes (kB):', handles=handles, labels=labels,
                   loc='upper center', ncol=2, fancybox=False, shadow=False)

    if save_fig:
        plt.tight_layout()
        plt.savefig(os.path.join(config['plots_dir_path'], f'{dataset_names_str}class_incr_learn_emulation={config["host"]}.pdf'), bbox_inches='tight', pad_inches=0.0)
    else:
        plt.show()

def plot_timing_measurements(config, dataset_name, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials, text_width, color_dict, save_fig=False):
    datetime_formats = ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']

    width_in, height_in = set_size(width=textwidth)
    fig, ax = plt.subplots(figsize=(width_in, height_in))

    x_labels = [r"\texttt{compute_dist()}"] + [rf"\texttt{{{func}()}}" if type(func) is str else rf"\texttt{{{func[0]}()}}" + f"\n($N_{{iter}}$={func[1]})" for func in sub_sel_funcs]
    x = np.arange(len(x_labels))
    width = 0.15 # the width of the bars

    compute_dist_time_list = []

    sub_sel_func_time_list = [[], [], []]

    sub_sel_func_time_mean = np.zeros(len(sub_sel_funcs))
    sub_sel_func_time_std = np.zeros(len(sub_sel_funcs))

    for color_idx, [ram_buffer_size, eeprom_buffer_size] in enumerate(buffer_sizes):
        label = f'({ram_buffer_size}, {eeprom_buffer_size})'

        for i, func in enumerate(sub_sel_funcs):
            for j in range(num_of_trials):
                for k, seq_type in enumerate(seq_types):
                    # filename_prefix = f'{dataset_name}_{func}_sub_selection_seq={seq_type}_trial={j+1}_'
                    exp_param = f'sub_selection_emulation={str(config["host"]).lower()}_seq={seq_type}_ram_buf_size={ram_buffer_size}_eeprom_buf_size={eeprom_buffer_size}_'
                    if func == 'rand_bal':
                        filename_prefix = f'{dataset_name}_{func}_' + exp_param + f'trial={j + 1}_'
                    elif func[0] == 'rand_greedy':
                        num_iter = func[1]
                        filename_prefix = f'{dataset_name}_{func[0]}_' + exp_param + f'num_iter={num_iter}_trial={j + 1}_'
                    elif func[0] == 'evo':
                        num_gen = func[1]
                        filename_prefix = f'{dataset_name}_{func[0]}_' + exp_param + f'num_gen={num_gen}_trial={j + 1}_'

                    tree = ET.parse(os.path.join(config['log_dir_path'], 'xml', filename_prefix + 'responses_log.xml'))
                    root = tree.getroot()

                    compute_dist_time_list += get_resp_elapsed_time(root, 'compute_dist_matrix', datetime_formats)

                    if func == 'rand_bal':
                        sub_sel_func_time_list[i] += get_resp_elapsed_time(root, 'rand_subset_selection', datetime_formats)
                    elif func[0] == 'rand_greedy' or func[0] == 'evo':
                        sub_sel_func_time_list[i] += get_resp_elapsed_time(root, func[0] + '_subset_selection', datetime_formats)

            sub_sel_func_time_mean[i] = np.mean(np.array(sub_sel_func_time_list[i]))
            sub_sel_func_time_std[i] = np.std(np.array(sub_sel_func_time_list[i]))

        compute_dist_time_mean = np.mean(np.array(compute_dist_time_list))
        compute_dist_time_std = np.std(np.array(compute_dist_time_list))

        height = [compute_dist_time_mean] + list(sub_sel_func_time_mean)
        yerr = [compute_dist_time_std] + list(sub_sel_func_time_std)

        ax.bar(x + (color_idx * width - (len(buffer_sizes) - 1) * width/2), height, width, label=label, color=color_dict[(ram_buffer_size, eeprom_buffer_size)])
        # ax.errorbar(x + (color_idx * width - (len(sub_sel_funcs) + 1) * width / 2), y=height, yerr=yerr, fmt='o', color='r')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, ha='center')
    ax.set_ylabel('Time (s)')
    ax.legend(title='Volatile and non-volatile\nmem. buffer sizes (kB):')
    ax.grid()
    ax.set_yscale('log')

    if save_fig:
        plt.tight_layout()
        plt.savefig(os.path.join(config['plots_dir_path'], f'func_time_emulation={config["host"]}.pdf'))
    else:
        plt.show()


if __name__ == '__main__':
    # Get configuration parameters
    config_dir_path = 'config/'
    config = read_config(config_dir_path, 'config_global.ini')

    # Create plots directory if it doesn't exist
    if not os.path.exists(config['plots_dir_path']):
        os.mkdir(config['plots_dir_path'])

    rc_params = {
        'figure.dpi': 300,
        'font.family': 'Linux Libertine O',
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.title_fontsize': 8,
        'text.usetex': True,
    }
    plt.rcParams.update(rc_params)

    buffer_sizes = [(32, 64),
                    (64, 128),
                    (256, 512),
                    (128, 256)]

    color_list = [mcolors.to_hex(cm.tab10(i / 7)) for i in range(8)]
    color_dict = {buffer_size : color_list[i] for i, buffer_size in enumerate(buffer_sizes)}

    textwidth = 395.8225
    dataset_names = ['MNIST', 'FashionMNIST']
    sub_sel_funcs = ['rand_bal', ['rand_greedy', 100], ['evo', 50]]
    seq_types = ['low', 'high']

    buffer_sizes = [(32, 64),
                    (64, 128),
                    # (128, 256),
                    (256, 512)]
    num_of_trials = 20
    plot_class_incr_learning_2(config, dataset_names, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials,
                               textwidth=textwidth, color_dict=color_dict, save_fig=True)

    buffer_sizes = [(32, 64),
                    (64, 128),
                    (128, 256),
                    (256, 512)]
    num_of_trials = 20
    plot_timing_measurements(config, dataset_names[0], sub_sel_funcs, seq_types, buffer_sizes, num_of_trials, textwidth, color_dict, save_fig=True)

    dataset_names = ['EMNIST']
    buffer_sizes = [(128, 256),
                    (256, 512)]
    plot_class_incr_learning_2(config, dataset_names, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials,
                               textwidth=textwidth, color_dict=color_dict, save_fig=True)
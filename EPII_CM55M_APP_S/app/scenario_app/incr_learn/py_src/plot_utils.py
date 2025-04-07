import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import os
import sys
import math
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


def get_command_resp(root, command_name):
    matching_resp = []
    for resp in root.findall('./response'):
        for child in resp:
            if child.text == command_name:
                matching_resp.append(resp)

    return matching_resp


def get_sub_sel_time(root, sub_sel_func):
    matching_resp = get_command_resp(root, sub_sel_func)

    sub_sel_time_list = []
    for resp in matching_resp:
        data_out_str = resp.find('data_out').text

        data_out_str = data_out_str.replace('array', 'np.array')
        data_out_str = data_out_str.replace('uint16', 'np.uint16')
        data_out_str = data_out_str.replace('uint8', 'np.uint8')

        data_out = eval(data_out_str)
        sub_sel_time = float(data_out[3][0]) * 1e-6 # Convert microseconds to seconds
        sub_sel_time_list.append(sub_sel_time)

    return sub_sel_time_list


def get_resp_elapsed_time(root, command_name, datetime_formats):
    matching_resp = get_command_resp(root, command_name)

    time_elapsed_list = []
    for resp in matching_resp:
        start_time_str = resp.find('start_time').text
        end_time_str = resp.find('end_time').text

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


def extract_line_plot_data(config, dataset_name, eval_metrics, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials):
    class_sequences = np.load(os.path.join(config['artifacts_dir_path'], dataset_name + '_class_sequences.npy'))
    class_seq_len = class_sequences.shape[1]

    data = np.zeros((len(eval_metrics), len(buffer_sizes), len(seq_types), len(sub_sel_funcs), 2, class_seq_len - 1))

    for col, seq_type in enumerate(seq_types):
        for row, eval_metric in enumerate(eval_metrics):
            for func_num, func in enumerate(sub_sel_funcs):
                for buffer_size_pair_num, (ram_buffer_size, eeprom_buffer_size) in enumerate(buffer_sizes):
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

                        with open(os.path.join(config['results_dir_path'], filename_prefix + 'results_dict.pkl'),
                                  'rb') as f:
                            results_dict = pickle.load(f)

                        if eval_metric != 'bwt':
                            metric[trial, :] = results_dict[eval_metric]
                        else:
                            metric[trial, :] = compute_backward_transfer(results_dict['acc_matrix'])

                    metric_mean = np.mean(metric, axis=0)
                    metric_std = np.std(metric, axis=0)

                    data[row, buffer_size_pair_num, col, func_num, 0, :] = metric_mean
                    data[row, buffer_size_pair_num, col, func_num, 1, :] = metric_std
    return data


def extract_time_measurements(config, dataset_name, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials):
    datetime_formats = ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']

    time_measurements = np.zeros((len(buffer_sizes), len(sub_sel_funcs), 2))

    for buffer_size_num, [ram_buffer_size, eeprom_buffer_size] in enumerate(buffer_sizes):
        compute_dist_time_list = []

        sub_sel_func_time_list = [[], [], []]
        sub_sel_func_time_median = np.zeros(len(sub_sel_funcs))
        sub_sel_func_time_iqr = np.zeros(len(sub_sel_funcs))

        for i, func in enumerate(sub_sel_funcs):
            for j in range(num_of_trials):
                for k, seq_type in enumerate(seq_types):
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
                        sub_sel_func_time_list[i] += get_sub_sel_time(root, 'rand_subset_selection')
                    elif func[0] == 'rand_greedy' or func[0] == 'evo':
                        sub_sel_func_time_list[i] += get_sub_sel_time(root, func[0] + '_subset_selection')

            sub_sel_func_time_median[i] = np.median(np.array(sub_sel_func_time_list[i]))
            q1 = np.percentile(np.array(sub_sel_func_time_list[i]), 25)
            q3 = np.percentile(np.array(sub_sel_func_time_list[i]), 75)
            sub_sel_func_time_iqr[i] = q3 - q1

            time_measurements[buffer_size_num, i, 0] = sub_sel_func_time_median[i]
            time_measurements[buffer_size_num, i, 1] = sub_sel_func_time_iqr[i]

    return time_measurements


def plot_class_incr_learning(config, dataset_names, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials, textwidth, color_dict, save_fig=False):
    eval_metrics = ['acc_test_set_union', 'acc_train_set_union']
    dataset_names_str = ''

    width_in, _ = set_size(width=textwidth, subplots=(len(eval_metrics), len(seq_types)*len(dataset_names)))

    fig, ax = plt.subplots(nrows=len(eval_metrics), ncols=len(seq_types)*len(dataset_names), figsize=(width_in, 2*2.0))

    for i, dataset_name in enumerate(dataset_names):
        class_sequences = np.load(os.path.join(config['artifacts_dir_path'], dataset_name + '_class_sequences.npy'))
        class_seq_len = class_sequences.shape[1]

        data = extract_line_plot_data(config, dataset_name, eval_metrics, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials)

        for col, seq_type in enumerate(seq_types):
            for row, eval_metric in enumerate(eval_metrics):
                for func_num, func in enumerate(sub_sel_funcs):
                    for buffer_size_pair_num, (ram_buffer_size, eeprom_buffer_size) in enumerate(buffer_sizes):
                        metric_mean = data[row, buffer_size_pair_num, col, func_num, 0, :]
                        metric_std = data[row, buffer_size_pair_num, col, func_num, 1, :]

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
                        ax[row, 2*i+col].plot(t, metric_mean, linestyle=linestyle, label=label, color=color_dict[(ram_buffer_size, eeprom_buffer_size)], lw=0.8)
                        ax[row, 2*i+col].fill_between(t, metric_mean - metric_std, metric_mean + metric_std,
                                                  color=color_dict[(ram_buffer_size, eeprom_buffer_size)], alpha=0.08, linestyle=linestyle, edgecolor='black', lw=0.8)

                        ax[row, 2*i+col].xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
                        ax[row, 2*i+col].grid(True, which='minor', alpha=0.3)
                        ax[row, 2*i+col].grid(True, which='major')

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
                                ax[row, 2*i+col].set_title('$\mathbf{q}_{\min}$')
                            elif seq_type == 'high':
                                ax[row, 2*i+col].set_title('$\mathbf{q}_{\max}$')

                        if col == 0 and i > 0:
                            ax[row, 2*i+col].set_yticklabels([])
                        if col > 0:
                            ax[row, 2*i+col].set_yticklabels([])

                        if not row == len(eval_metrics) - 1:
                            ax[row, 2*i+col].set_xticklabels([])

                        if dataset_name == 'EMNIST':
                            ax[row, 2 * i + col].set_xticks([2, 14, 26, 38, 47])
                        else:
                            ax[row, 2*i+col].set_xticks([i for i in range(2, class_seq_len+1, int(class_seq_len / 5))])

        if dataset_name == 'MNIST' or dataset_name == 'EMNIST':
            for j, eval_metric in enumerate(eval_metrics):
                if eval_metric == 'acc_test_set_union':
                    ax[j, 0].set_ylabel('$A_{1}$ on test set $\mathcal{T}_{t}$')
                elif eval_metric == 'acc_train_set_union':
                    ax[j, 0].set_ylabel('$A_{1}$ on train set $\{\mathcal{B}_{t}\}_{i=1}^{t}$')

        ax[1, 2*i].set_xlabel('Num of classes')
        ax[1, 2*i+1].set_xlabel('Num of classes')

        dataset_names_str += dataset_name + '_'
    # handles, labels = plt.gca().get_legend_handles_labels()

    if len(dataset_names) == 2:
        plt.figtext(0.30, 0.96, dataset_names[0], va="center", ha="center")
        plt.figtext(0.75, 0.96, dataset_names[1], va="center", ha="center")
    #     fig.legend(title='Volatile and non-volatile mem. buffer sizes (kB):', handles=handles, labels=labels,
    #                loc='upper center', bbox_to_anchor=(0.5, 0.03),
    #                ncol=4, fancybox=False, shadow=False)
    else:
        plt.figtext(0.53, 0.96, dataset_names[0], va="center", ha="center")

        buffer_sizes = [(32, 64),
                        (128, 256),
                        (64, 128),
                        (256, 512)]

        colors = [color_dict[buffer_size] for buffer_size in buffer_sizes]
        # Plot dummy lines
        lines = [ax[0, 0].plot([], [], color=color)[0] for color in colors]

        ax[0, 0].legend(lines, buffer_sizes, title='Volatile and non-volatile mem.\nbuffer sizes (kB):',
                   loc='upper center', ncol=2, fancybox=False, shadow=False)

        # ax[0, 0].legend(title='Volatile and non-volatile mem.\nbuffer sizes (kB):', handles=handles, labels=labels,
        #            loc='upper center', ncol=2, fancybox=False, shadow=False)

    if save_fig:
        plt.tight_layout()
        plt.savefig(os.path.join(config['plots_dir_path'], f'{dataset_names_str}class_incr_learn_emulation={config["host"]}.pdf'), bbox_inches='tight', pad_inches=0.0)
    else:
        plt.show()


def plot_timing_measurements(config, dataset_name, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials, textwidth, color_dict, save_fig=False):
    datetime_formats = ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']

    width_in, height_in = set_size(width=textwidth)
    fig, ax = plt.subplots(figsize=(width_in, height_in))

    # x_labels = [r"\texttt{compute_dist()}"] + [rf"\texttt{{{func}()}}" if type(func) is str else rf"\texttt{{{func[0]}()}}" + f"\n($N_{{iter}}$={func[1]})" for func in sub_sel_funcs]
    x_labels = [rf"\texttt{{{func}()}}" if type(func) is str else rf"\texttt{{{func[0]}()}}" + f"\n($N_{{iter}}$={func[1]})" for func in sub_sel_funcs]
    x = np.arange(len(x_labels))
    width = 0.15 # the width of the bars


    for color_idx, [ram_buffer_size, eeprom_buffer_size] in enumerate(buffer_sizes):
        label = f'({ram_buffer_size}, {eeprom_buffer_size})'

        compute_dist_time_list = []

        sub_sel_func_time_list = [[] for _ in range(0, len(sub_sel_funcs))]
        sub_sel_func_time_median = np.zeros(len(sub_sel_funcs))
        sub_sel_func_time_iqr = np.zeros(len(sub_sel_funcs))

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
                        sub_sel_func_time_list[i] += get_sub_sel_time(root, 'rand_subset_selection')

                    elif func[0] == 'rand_greedy' or func[0] == 'evo':
                        sub_sel_func_time_list[i] += get_sub_sel_time(root, func[0] + '_subset_selection')

            sub_sel_func_time_median[i] = np.median(np.array(sub_sel_func_time_list[i]))
            q1 = np.percentile(np.array(sub_sel_func_time_list[i]), 25)
            q3 = np.percentile(np.array(sub_sel_func_time_list[i]), 75)
            sub_sel_func_time_iqr[i] = q3 - q1

        # compute_dist_time_mean = np.mean(np.array(compute_dist_time_list))
        # compute_dist_time_std = np.std(np.array(compute_dist_time_list))

        # height = [compute_dist_time_mean] + list(sub_sel_func_time_mean)
        # yerr = [compute_dist_time_std] + list(sub_sel_func_time_std)

        x_box_plot = sub_sel_func_time_list
        positions = x + (color_idx * width - (len(buffer_sizes) - 1) * width/2)
        box = ax.boxplot(x=x_box_plot, positions=positions, showmeans=True, showfliers=True, widths=width*0.9, label=label)

        for median_num, median in enumerate(box['medians']):
            median.set_color(color_dict[(ram_buffer_size, eeprom_buffer_size)])  # Change to any color you like

        # ax.bar(x + (color_idx * width - (len(buffer_sizes) - 1) * width/2), height, width, label=label, color=color_dict[(ram_buffer_size, eeprom_buffer_size)])
        # ax.errorbar(x + (color_idx * width - (len(buffer_sizes) - 1) * width / 2), y=height, yerr=yerr, fmt='o', color='r')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, ha='center')
    ax.set_ylabel('Time (s)')
    ax.legend(title='Volatile and non-volatile\nmem. buffer sizes (kB):')
    ax.grid()
    ax.set_yscale('log')

    if save_fig:
        plt.tight_layout()
        plt.savefig(os.path.join(config['plots_dir_path'], f'{dataset_name}_func_time_emulation={config["host"]}.pdf'))
    else:
        plt.show()


def class_incr_acc_table(config, dataset_name, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials):
    eval_metrics = ['acc_test_set_union', 'acc_train_set_union']

    class_sequences = np.load(os.path.join(config['artifacts_dir_path'], dataset_name + '_class_sequences.npy'))
    class_seq_len = class_sequences.shape[1]
    data = extract_line_plot_data(config, dataset_name, eval_metrics, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials)

    offset = 2
    if dataset_name == 'EMNIST':
        num_of_classes = [14, 26, 38, 47]
        data_idxs = [i - offset for i in num_of_classes]
    else:
        num = 4
        data_idxs = [math.ceil( (class_seq_len - offset) * i/num) for i in range(1, num+1)]
        num_of_classes = [i + offset for i in data_idxs]

    with open(os.path.join(config['plots_dir_path'], f'{dataset_name}_acc_table.txt'), 'w') as f:
        print(f'{dataset_name}, num_of_classes: {num_of_classes}', file=f)

        for eval_metric_num, eval_metric in enumerate(eval_metrics):
            if eval_metric == 'acc_test_set_union':
                metric_str = f'\multirow{{{len(buffer_sizes) * len(seq_types)}}}{{*}}{{\\rotatebox{{90}}{{Test set $A_{{1}}$}}}} & '
            else:
                metric_str = f'\multirow{{{len(buffer_sizes) * len(seq_types)}}}{{*}}{{\\rotatebox{{90}}{{Training set $A_{{1}}$}}}} & '
            print(metric_str, end='', file=f)

            for buffer_size_pair_num, (ram_buffer_size, eeprom_buffer_size) in enumerate(buffer_sizes):
                if buffer_size_pair_num == 0:
                    f.seek(0, 2)  # Move to end of file
                    f.seek(f.tell() - 2)  # Move back 2 characters
                    f.truncate()  # Remove the last 2 characters
                    # print('\b\b', end='')

                for seq_num, seq_type in enumerate(seq_types):

                    if seq_num == 0:
                        print(f'& \multirow{{2}}{{*}}{{({ram_buffer_size}, {eeprom_buffer_size})}} & ', end='', file=f)
                    else:
                        print('& &', end='', file=f)

                    if seq_type == 'high':
                        print('$\mathbf{q}_{\max}$ & ', end='', file=f)
                    else:
                        print('$\mathbf{q}_{\min}$ & ', end='', file=f)

                    for i in data_idxs:
                        for func_num, func in enumerate(sub_sel_funcs):
                            metric_mean = data[eval_metric_num, buffer_size_pair_num, seq_num, func_num, 0, i]
                            metric_mean *= 100

                            if func == 'rand_bal':
                                print(f'{metric_mean:.2f} & ', end='', file=f)
                            else:
                                rand_bal_metric_mean = data[eval_metric_num, buffer_size_pair_num, seq_num, 0, 0, i]
                                diff = metric_mean - rand_bal_metric_mean * 100
                                print(f'{diff:.2f} & ', end='', file=f)

                    f.seek(0, 2)  # Move to end of file
                    f.seek(f.tell() - 2)  # Move back 2 characters
                    f.truncate()  # Remove the last 2 characters
                    print('\\\\', file=f)

                    if seq_num == 0:
                        print('\cline{3-15}', file=f)
                    else:
                        print('\cline{2-15}', file=f)
            print('\hline\hline', file=f)
        print('', end='\n\n', file=f)
    sys.stdout = sys.__stdout__


def timing_measurements_table(config, dataset_name, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials):

    hpc_time_measurements = extract_time_measurements(config, dataset_name, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials)

    # Set 'host' flag false to get time measurements from target board
    config['host'] = False
    num_of_trials = 2
    dev_time_measurements = extract_time_measurements(config, dataset_name, sub_sel_funcs, seq_types, buffer_sizes,  num_of_trials)

    # Reset flag to True
    config['host'] = True

    dev_types = ['HPC', 'Edge']

    with open(os.path.join(config['plots_dir_path'], f'latency_table.txt'), 'w') as f:

        for buffer_size_pair_num, buffer_size_pair in enumerate(buffer_sizes):
            for dev_num, dev_type in enumerate(dev_types):

                if dev_num == 0:
                    print(f'\multirow{{2}}{{*}}{{{buffer_size_pair}}} & ', end='', file=f)
                else:
                    print('& ', end='', file=f)

                print(f'{dev_type} & ', end='', file=f)

                for func_num, _ in enumerate(sub_sel_funcs):

                    if dev_type == 'HPC':
                        print(f"{hpc_time_measurements[buffer_size_pair_num, func_num, 0]:.2e} & & {hpc_time_measurements[buffer_size_pair_num, func_num, 1]:.2e} & ", end='', file=f)
                    else:
                        multiplier = dev_time_measurements[buffer_size_pair_num, func_num, 0] / hpc_time_measurements[buffer_size_pair_num, func_num, 0]
                        print(rf"{dev_time_measurements[buffer_size_pair_num, func_num, 0]:.2e} & \text{{$(\times\;${multiplier:1.1f})}} & {dev_time_measurements[buffer_size_pair_num, func_num, 1]:.2e} & ", end='', file=f)

                # Delete last two characters
                f.seek(0, 2)  # Move to end of file
                f.seek(f.tell() - 2)  # Move back 2 characters
                f.truncate()  # Remove the last 2 characters
                print('\\\\', file=f)
                print('\cline{2-11}', file=f) if dev_num == 0 else print('\hline\hline', file=f)


def plot_acc_time_pareto_front(config, dataset_name, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials, textwidth, color_dict, save_fig=False):
    eval_metrics = ['acc_test_set_union', 'acc_train_set_union']

    class_sequences = np.load(os.path.join(config['artifacts_dir_path'], dataset_name + '_class_sequences.npy'))
    class_seq_len = class_sequences.shape[1]
    acc_data = extract_line_plot_data(config, dataset_name, eval_metrics, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials)

    offset = 2
    if dataset_name == 'EMNIST':
        num_of_classes = [14, 26, 38, 47]
        data_idxs = [i - offset for i in num_of_classes]
    else:
        num = 4
        data_idxs = [math.ceil( (class_seq_len - offset) * i/num) for i in range(1, num+1)]
        num_of_classes = [i + offset for i in data_idxs]

    time_measurements = extract_time_measurements(config, dataset_name, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials)

    width_in, height_in = set_size(width=textwidth, subplots=(len(eval_metrics), len(num_of_classes)))
    fig, ax = plt.subplots(nrows=len(eval_metrics), ncols=len(num_of_classes), figsize=(width_in, 4))

    func_marker_dict = {'rand_bal': 'o', 'rand_greedy': '^', 'evo': 's'}

    for eval_metric_num, eval_metric in enumerate(eval_metrics):
        for col, data_idx in enumerate(data_idxs):
            for buffer_size_pair_num, buffer_size_pair in enumerate(buffer_sizes):
                t_list = []
                acc_list = []
                color = color_dict[buffer_size_pair]

                for func_num, func in enumerate(sub_sel_funcs):
                    marker = func_marker_dict[func[0]] if isinstance(func, list) else func_marker_dict[func]

                    t = time_measurements[buffer_size_pair_num, func_num, 0]
                    t_list.append(t)

                    # Take the mean of the top-1 acc between q_max and q_min
                    acc = np.mean(acc_data[eval_metric_num, buffer_size_pair_num, :, func_num, 0, data_idx])
                    acc_list.append(acc)

                    ax[eval_metric_num, col].scatter(t, acc, marker=marker, s=10, color=color)

                ax[eval_metric_num, col].plot(t_list, acc_list, color=color, linestyle='--', linewidth=0.5,
                                              label=str(buffer_size_pair))

            ax[eval_metric_num, col].set_xscale('log')
            ax[eval_metric_num, col].set_xlim([2e-6, 1e3])
            major_ticks = [math.pow(10, exp) for exp in range(-5, 5, 2)]
            ax[eval_metric_num, col].set_xticks(major_ticks)

            minor_ticks = [math.pow(10, exp) for exp in range(-4, 4, 2)]
            ax[eval_metric_num, col].xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))

            ax[eval_metric_num, col].tick_params(axis='x', which='major', labelsize=5)
            ax[eval_metric_num, col].tick_params(axis='x', which='minor', labelbottom=False)

            ax[eval_metric_num, col].yaxis.minorticks_on()
            ax[eval_metric_num, col].grid(True, which='minor', alpha=0.3)
            ax[eval_metric_num, col].grid(True, which='major')


            if dataset_name == 'MNIST':
                ax[eval_metric_num, col].set_ylim([0.65, 1.0])
            elif dataset_name == 'FashionMNIST':
                ax[eval_metric_num, col].set_ylim([0.5, 0.9])
            else:
                ax[eval_metric_num, col].set_ylim([0.3, 1.0])


            if col > 0:
                ax[eval_metric_num, col].set_yticklabels([])

            if eval_metric_num == 0:
                ax[eval_metric_num, col].set_title(f'{num_of_classes[col]} classes')

            if not eval_metric_num == len(eval_metrics) - 1:
                ax[eval_metric_num, col].set_xticklabels([])
            else:
                ax[eval_metric_num, col].set_xlabel('time ($s$)')

            if eval_metric == 'acc_test_set_union':
                ax[eval_metric_num, 0].set_ylabel('$A_{1}$ on test set $\mathcal{T}_{t}$')
            elif eval_metric == 'acc_train_set_union':
                ax[eval_metric_num, 0].set_ylabel('$A_{1}$ on train set $\{\mathcal{B}_{t}\}_{i=1}^{t}$')

    if dataset_name == 'MNIST':
        colors = [color_dict[buffer_size] for buffer_size in buffer_sizes[::-1]]
        # Plot dummy lines
        lines = [ax[1, 0].plot([], [], color=color, linestyle='--', linewidth=0.8)[0] for color in colors]
        ax[1, 0].legend(lines, buffer_sizes[::-1], title='Volatile and non-\nvolatile buf. (kB):',
                        loc='lower center', fancybox=False, shadow=False, fontsize=6)

        # Dummy scatter plots
        func_names = [f"$\\texttt{{{func_name}()}}$" for func_name in list(func_marker_dict.keys())]
        scatter = [ax[1, 1].scatter([], [], marker=func_marker_dict[func_name], s=10, color='black') for func_name in list(func_marker_dict.keys())]
        ax[1, 1].legend(scatter, func_names, title='Sub. sel. funcs.', fontsize=6)

        fig.suptitle(f'{dataset_name}')

    if save_fig:
        plt.tight_layout()
        plt.savefig(os.path.join(config['plots_dir_path'], f'{dataset_name}_pareto_front.pdf'), bbox_inches='tight', pad_inches=0.0)
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

    config['host'] = True
    textwidth = 395.8225
    sub_sel_funcs = ['rand_bal', ['rand_greedy', 100], ['evo', 50]]
    seq_types = ['low', 'high']
    num_of_trials = 20

    # Get MNIST and FashionMNIST plots ---------------------------------------------------------------------------------
    print('Creating MNIST and FashionMNIST plots and tables...')
    dataset_names = ['MNIST', 'FashionMNIST']
    buffer_sizes = [(32, 64),
                    (64, 128),
                    # (128, 256),
                    (256, 512)]
    plot_class_incr_learning(config, dataset_names, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials,
                               textwidth=textwidth, color_dict=color_dict, save_fig=True)

    seq_types = ['high', 'low'] # Reverse seq order for table
    buffer_sizes = [(32, 64),
                    (64, 128),
                    (128, 256),
                    (256, 512)]
    for dataset_name in dataset_names:
        class_incr_acc_table(config, dataset_name, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials)
        plot_timing_measurements(config, dataset_name, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials,
                                textwidth, color_dict, save_fig=True)
        plot_acc_time_pareto_front(config, dataset_name, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials,
                                textwidth, color_dict, save_fig=True)

    timing_measurements_table(config, 'MNIST', sub_sel_funcs, seq_types, buffer_sizes, num_of_trials)

    # Get EMNIST plots -------------------------------------------------------------------------------------------------
    print('Creating EMNIST plots and tables...')
    dataset_names = ['EMNIST']
    seq_types = ['low', 'high']
    buffer_sizes = [(128, 256),
                    (256, 512)]
    plot_class_incr_learning(config, dataset_names, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials,
                               textwidth=textwidth, color_dict=color_dict, save_fig=True)

    seq_types = ['high', 'low'] # Reverse seq order for table
    class_incr_acc_table(config, 'EMNIST', sub_sel_funcs, seq_types, buffer_sizes, num_of_trials)

    plot_acc_time_pareto_front(config, 'EMNIST', sub_sel_funcs, seq_types, buffer_sizes, num_of_trials,
                               textwidth, color_dict, save_fig=True)


import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
import xml.etree.ElementTree as ET
from datetime import datetime as dt

from util_functions import read_config


def get_resp_elapsed_time(root, command_name, datetime_format):
    matching_req = []
    for req in root.findall('./response'):
        for child in req:
            if child.text == command_name:
                matching_req.append(req)

    time_elapsed_list = []
    for req in matching_req:
        start_time_str = req.find('start_time').text
        end_time_str = req.find('end_time').text

        start_time = dt.strptime(start_time_str, datetime_format)
        end_time = dt.strptime(end_time_str, datetime_format)

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


def plot_class_incr_learning_2(config, dataset_name, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials, save_fig=False):
    eval_metrics = ['acc_test_set_union', 'acc_global']
    class_sequences = np.load(os.path.join(config['artifacts_dir_path'], dataset_name + '_class_sequences.npy'))
    class_seq_len = class_sequences.shape[1]

    fig, ax = plt.subplots(nrows=len(eval_metrics), ncols=len(seq_types), figsize=(8, 10))
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for col, seq_type in enumerate(seq_types):
        for row, eval_metric in enumerate(eval_metrics):
            for func in sub_sel_funcs:
                for color_idx, [ram_buffer_size, eeprom_buffer_size] in enumerate(buffer_sizes):
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
                        label = f'{ram_buffer_size}, {eeprom_buffer_size}'
                    elif func[0] == 'rand_greedy':
                        linestyle = '--'
                        label = None
                    elif func[0] == 'evo':
                        linestyle = 'dotted'
                        label = None

                    t = np.arange(2, class_seq_len + 1)
                    ax[row, col].plot(t, metric_mean, linestyle=linestyle, label=label, color=color_list[color_idx])
                    ax[row, col].fill_between(t, metric_mean - metric_std, metric_mean + metric_std,
                                              color=color_list[color_idx], alpha=0.1)
                    ax[row, col].grid(True)

                    if eval_metric == 'acc_test_set_union':
                        if dataset_name == 'FashionMNIST' or dataset_name == 'MNIST':
                            ax[row, col].set_ylim([0.55, 1.0])
                        else:
                            ax[row, col].set_ylim([0.1, 1.0])
                    elif eval_metric == 'acc_global':
                        ax[row, col].set_ylim([0, 1.0])

                    if row == 0:
                        ax[row, col].set_title(f'{seq_type} acc seq')

                    if col > 0:
                        ax[row, col].set_yticklabels([])



    ax[0, 0].legend(title='N_RAM, N_EEPROM')
    ax[0, 0].set_ylabel('ACC test set union')
    ax[1, 0].set_ylabel('ACC global')
    # ax[2, 0].set_ylabel('Backward TF')

    ax[1, 0].set_xlabel('Num of classes')
    ax[1, 1].set_xlabel('Num of classes')

    fig.suptitle(f"{dataset_name}, N_Trials={num_of_trials}")

    if save_fig:
        plt.savefig(os.path.join(config['plots_dir_path'], f'{dataset_name}_class_incr_learn_emulation={config["host"]}_n_trials={num_of_trials}.pdf'))
    else:
        plt.show()

def plot_timing_measurements(config, dataset_name, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials, save_fig=False):
    class_sequences = np.load(os.path.join(config['artifacts_dir_path'], dataset_name + '_class_sequences.npy'))
    class_seq_len = class_sequences.shape[1]

    datetime_format = '%Y-%m-%d %H:%M:%S.%f'

    fig, ax = plt.subplots()

    x_labels = ['compute_dist()'] + [func + '()' if type(func) is str else f'{func[0]}()\n(num_iter={func[1]})' for func in sub_sel_funcs]
    x = np.arange(len(x_labels))
    width = 0.15 # the width of the bars

    compute_dist_time = np.zeros((len(sub_sel_funcs), len(seq_types) * num_of_trials, class_seq_len - 1))

    sub_sel_func_time = np.zeros((len(sub_sel_funcs), len(seq_types) * num_of_trials, class_seq_len - 1))
    sub_sel_func_time_mean = np.zeros(len(sub_sel_funcs))
    sub_sel_func_time_std = np.zeros(len(sub_sel_funcs))

    for color_idx, [ram_buffer_size, eeprom_buffer_size] in enumerate(buffer_sizes):
        label = f'{ram_buffer_size}, {eeprom_buffer_size}'

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

                    compute_dist_time[i, k*num_of_trials + j, :] = np.array(get_resp_elapsed_time(root, 'compute_dist_matrix', datetime_format))

                    if func == 'rand_bal':
                        sub_sel_func_time[i, k*num_of_trials + j, :] = np.array(get_resp_elapsed_time(root, 'rand_subset_selection', datetime_format))
                    elif func[0] == 'rand_greedy' or func[0] == 'evo':
                        sub_sel_func_time[i, k*num_of_trials + j, :] = np.array(get_resp_elapsed_time(root, func[0] + '_subset_selection', datetime_format))

            sub_sel_func_time_mean[i] = np.mean(sub_sel_func_time[i, :, :])
            sub_sel_func_time_std[i] = np.std(sub_sel_func_time[i, :, :])

        compute_dist_time_mean = np.mean(compute_dist_time)
        compute_dist_time_std = np.std(compute_dist_time)

        height = [compute_dist_time_mean] + list(sub_sel_func_time_mean)
        yerr = [compute_dist_time_std] + list(sub_sel_func_time_std)

        ax.bar(x + (color_idx * width - (len(buffer_sizes) - 1) * width/2), height, width, label=label)
        # ax.errorbar(x + (color_idx * width - (len(sub_sel_funcs) + 1) * width / 2), y=height, yerr=yerr, fmt='o', color='r')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Time (s)')
    ax.legend(title='N_RAM, N_EEPROM')
    ax.grid()
    ax.set_yscale('log')
    fig.suptitle(f"{dataset_name}, N_Trials={num_of_trials}")

    if save_fig:
        plt.savefig(os.path.join(config['plots_dir_path'], f'{dataset_name}_func_time_emulation={config["host"]}_n_trials={num_of_trials}.pdf'))
    else:
        plt.show()


if __name__ == '__main__':
    # Get configuration parameters
    config_dir_path = 'config/'
    config = read_config(config_dir_path, 'config_global.ini')

    dataset_names = ['FashionMNIST', 'MNIST']
    sub_sel_funcs = ['rand_bal', ['rand_greedy', 100], ['evo', 50]]
    seq_types = ['low', 'high']

    buffer_sizes = [[50, 100],
                    [100, 200],
                    [200, 400],
                    [400, 800]]

    num_of_trials = 5

    for dataset_name in dataset_names:
        # plot_class_incr_learning(config, dataset_name, sub_sel_funcs, seq_types, ram_buffer_size, eeprom_buffer_size,
        #                          num_of_trials)
        plot_class_incr_learning_2(config, dataset_name, sub_sel_funcs, seq_types, buffer_sizes, num_of_trials)
        plot_timing_measurements(config, dataset_name, sub_sel_funcs, seq_types, buffer_sizes,
                                 num_of_trials)
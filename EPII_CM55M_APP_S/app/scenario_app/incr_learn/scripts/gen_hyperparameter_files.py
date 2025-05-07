import os
import sys
import pickle
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from py_src.util_functions import read_config

if __name__ == '__main__':
    # Get configuration parameters
    config_dir_path = 'config/'
    config = read_config(config_dir_path, 'config_global.ini')
    config |= read_config(config_dir_path, 'config_exper.ini')

    sub_sel_funcs = ['greedy', 'evo']

    mutation_rate_list = [0.01, 0.05, 0.10, 0.20]
    population_size_list = [20, 40, 60, 80]
    num_of_parents_list = [10, 20, 30]

    param_combinations = [list(combination) for combination in product(population_size_list, num_of_parents_list, mutation_rate_list)]

    for i, mutation_rate in enumerate(mutation_rate_list):
        with open(os.path.join(config['artifacts_dir_path'], f'greedy_hyper{i+1:02}.pkl'), 'wb') as f:
            hyperparameter_list = [config['num_iter'], mutation_rate]
            pickle.dump(hyperparameter_list, f)
    print(f'Greedy hyperparam files: {i+1}')

    for i, param_combination in enumerate(param_combinations):
        with open(os.path.join(config['artifacts_dir_path'], f'evo_hyper{i+1:02}.pkl'), 'wb') as f:
            hyperparameter_list = [config['num_gen']] + param_combination
            pickle.dump(hyperparameter_list, f)
    print(f'Evo hyperparam files: {i+1}')

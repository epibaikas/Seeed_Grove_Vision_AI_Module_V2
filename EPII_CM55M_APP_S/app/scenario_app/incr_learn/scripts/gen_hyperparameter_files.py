import os
import sys
import pandas as pd
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from py_src.util_functions import read_config

if __name__ == '__main__':
    # Get configuration parameters
    config_dir_path = 'config/'
    config = read_config(config_dir_path, 'config_global.ini')
    config |= read_config(config_dir_path, 'config_exper.ini')

    sub_sel_funcs = ['greedy', 'evo']

    num_iter = [5000]
    num_gen = [50, 100, 200]
    population_size_list = [100, 50, 25]
    num_gen_population_size_pairs = [list(pair) for pair in zip(num_gen, population_size_list)]

    num_of_parents_list = [5, 10, 15, 20]
    mutation_rate_list = [0.01, 0.05, 0.10, 0.20]

    greedy_param_combinations = [list(combination) for combination in product(num_iter, mutation_rate_list)]
    evo_param_combinations = [list(combination) for combination in product(num_gen_population_size_pairs, num_of_parents_list, mutation_rate_list)]

    dtype_spec = {
        'sub_sel_func': str,
        'hyperparam_file': str,
        'num_iter': pd.Int64Dtype(),
        'num_gen': pd.Int64Dtype(),
        'population_size': pd.Int64Dtype(),
        'num_parents': pd.Int64Dtype(),
        'mutation_rate': float
    }

    # Check if index for hyperparameter files exists
    if os.path.exists(os.path.join(config['artifacts_dir_path'], config['hyperparam_index_file_name'])):
        hyperparam_index_pd = pd.read_csv(os.path.join(config['artifacts_dir_path'],  config['hyperparam_index_file_name']),
                                            dtype=dtype_spec)
        print('Loaded index for hyperparameter sets.')
    else:
        # Create Pandas dataframe to be used as index for the hyperparameter files
        hyperparam_index_pd = pd.DataFrame(columns=['sub_sel_func', 'hyperparam_set', 'num_iter', 'num_gen',
                                                      'population_size', 'num_parents', 'mutation_rate'])
        print('Index for hyperparameter sets does not exist.')


    for i, greedy_param_combination in enumerate(greedy_param_combinations):
        hyperparam_set_name = f'greedy_hyper{i+1:02}'
        new_row = ['greedy', hyperparam_set_name, greedy_param_combination[0], 0, 0, 0, greedy_param_combination[1]]

        if not (hyperparam_index_pd == new_row).all(1).any():
            new_df = pd.DataFrame([new_row], columns=hyperparam_index_pd.columns)
            if hyperparam_index_pd.empty:
                hyperparam_index_pd = new_df
            else:
                hyperparam_index_pd = pd.concat([hyperparam_index_pd, new_df], ignore_index=True)

    print(f'Greedy hyperparam sets: {i+1}')

    for i, evo_param_combination in enumerate(evo_param_combinations):
        hyperparam_set_name = f'evo_hyper{i+1:02}'
        new_row = ['evo', hyperparam_set_name, 0, evo_param_combination[0][0], evo_param_combination[0][1],
                                                    evo_param_combination[1], evo_param_combination[2]]

        if not ((hyperparam_index_pd == new_row).all(axis=1)).any():
            new_df = pd.DataFrame([new_row], columns=hyperparam_index_pd.columns)
            if hyperparam_index_pd.empty:
                hyperparam_index_pd = new_df
            else:
                hyperparam_index_pd = pd.concat([hyperparam_index_pd, new_df], ignore_index=True)

    print(f'Evo hyperparam sets: {i+1}')

    hyperparam_index_pd.to_csv(os.path.join(config['artifacts_dir_path'], config['hyperparam_index_file_name']), index=False)

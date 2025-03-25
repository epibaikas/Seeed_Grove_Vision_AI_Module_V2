import concurrent.futures
import subprocess
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from py_src.argparse_utils import *

def run_trial(script_args):
    script, args = script_args
    subprocess.run(['python', script] + args, capture_output=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script for running class-incremental learning experiments')

    parser.add_argument('dataset', type=str, help='The name of the dataset to be used')
    parser.add_argument('ram_buf_size', type=positive_int,
                        help='The size of RAM buffer given in KBs')
    parser.add_argument('eeprom_buf_size', type=positive_int,
                        help='The size of the EEPROM buffer given in KBs')

    args = vars(parser.parse_args())

    # Get the arguments
    dataset_name = args['dataset']
    ram_buf_size = args['ram_buf_size']
    eeprom_buf_size = args['eeprom_buf_size']

    dataset_names = ['FashionMNIST', 'MNIST', 'EMNIST']
    if dataset_name not in dataset_names:
        raise ValueError(f'Not valid dataset name {dataset_name}')

    if dataset_name == 'EMNIST':
        MAX_WORKERS = 6
    else:
        MAX_WORKERS = 12

    py_file_path = 'py_src/sub_selection.py'

    scripts_with_args = []
    start_trial = 1
    num_of_trials = 20
    sub_sel_funcs = [1, 2, 3]
    seq_types = ['low', 'high']

    for func in sub_sel_funcs:
        for trial in range(start_trial, num_of_trials + 1):
            for seq_type in seq_types:
                scripts_with_args.append((py_file_path,  [dataset_name, str(func), seq_type,
                                                          str(ram_buf_size), str(eeprom_buf_size), str(trial)]))

    # Use ProcessPoolExecutor to limit concurrent execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_script = {executor.submit(run_trial, script_args): script_args for script_args in scripts_with_args}

        for future in concurrent.futures.as_completed(future_to_script):
                script = future_to_script[future]
                print(f'{script[0]} with parameters {script[1]} completed!')

                # try:
                #     output = future.result()
                #     print(f'Output from {script}:\n{output}')
                # except Exception as e:
                #     print(f'Error running {script}: {e}')
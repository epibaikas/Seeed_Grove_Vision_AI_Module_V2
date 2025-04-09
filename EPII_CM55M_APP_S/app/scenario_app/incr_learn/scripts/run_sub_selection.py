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
    parser.add_argument('sub_sel_func', type=str, help='Subset selection function (\'rand\' for random '
                                                       'selection, \'greedy\' for '
                                                       'greedy selection and \'evo\' for evolutionary)')
    parser.add_argument('bal', type=int, help='Enter \'1\' for maintaining class balancing in '
                                                       'subsets, \'0\' otherwise')
    parser.add_argument('ram_buf_size', type=positive_int,
                        help='The size of RAM buffer given in KBs')
    parser.add_argument('eeprom_buf_size', type=positive_int,
                        help='The size of the EEPROM buffer given in KBs')
    parser.add_argument('--target_dev', action='store_false', help='Use --target_dev flag when experiments will be run on the actual device')

    args = vars(parser.parse_args())

    # Get the arguments
    dataset_name = args['dataset']
    sub_sel_func = args['sub_sel_func']
    bal = args['bal']
    ram_buf_size = args['ram_buf_size']
    eeprom_buf_size = args['eeprom_buf_size']
    host = args['target_dev']

    print(f'host={host}')

    dataset_names = ['FashionMNIST', 'MNIST', 'EMNIST']
    if dataset_name not in dataset_names:
        raise ValueError(f'Not valid dataset name {dataset_name}')

    if bal not in [0, 1]:
        raise argparse.ArgumentTypeError('Invalid class balancing argument')

    sub_sel_funcs = ['rand', 'greedy', 'evo']
    if sub_sel_func not in sub_sel_funcs:
        raise ValueError(f'Not valid subset selection function name {sub_sel_func}')

    py_file_path = 'py_src/sub_selection.py'

    if host:
        if dataset_name == 'EMNIST':
            MAX_WORKERS = 10
        else:
            MAX_WORKERS = 12

        scripts_with_args = []
        start_trial = 1
        num_of_trials = 20
        seq_types = ['low', 'high']

        for trial in range(start_trial, num_of_trials + 1):
            for seq_type in seq_types:
                scripts_with_args.append((py_file_path,  [dataset_name, sub_sel_func, str(bal), seq_type,
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
    else:
        start_trial = 1
        num_of_trials = 2
        seq_types = ['low', 'high']

        for trial in range(start_trial, num_of_trials + 1):
            for seq_type in seq_types:
                args = [dataset_name, sub_sel_func, str(bal), seq_type, str(ram_buf_size),
                        str(eeprom_buf_size), str(trial), '--target_dev']
                
                process = subprocess.Popen(["python", py_file_path] + args, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, text=True, bufsize=1)

                # Read and print output in real-time
                for line in process.stdout:
                    print(line, end="")  # Print each line as it comes

                # Wait for the process to finish
                process.wait()
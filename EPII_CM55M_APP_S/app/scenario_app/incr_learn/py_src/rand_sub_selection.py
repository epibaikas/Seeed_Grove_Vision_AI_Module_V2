import argparse

from protocol_functions import *
from argparse_utlis import *
from util_functions import *
from data_utils import load_dataset, get_class_example_indices

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser('Script for running class-incremental learning via random data subset selection '
                                     'experiments on Seeed Grove Vision AI Module V2')

    parser.add_argument('dataset', type=str, help='The name of the dataset to be used')
    parser.add_argument('balanced', type=positive_int, help='Balanced subset option')
    parser.add_argument('trial', type=positive_int,
                        help='The experiment trial number used to adjust random seed for random sampling functions')

    args = vars(parser.parse_args())

    # Get the arguments
    dataset_name = args['dataset']
    balanced = args['balanced']
    trial = args['trial']

    # Get configuration parameters
    config_dir_path = 'config/'
    config = read_config(config_dir_path)

    # Adjust the random seed based on the trial number
    random_seed = config['random_seed'] + trial
    np.random.seed(random_seed)

    # Load dataset
    device = 'cpu'
    train_set, test_set, X_train, y_train, X_test, y_test = load_dataset(dataset_name, device)

    train_data = np.zeros(shape=(len(train_set), config['bytes_per_img']), dtype=np.uint8)
    train_data[:, 0:config['data_bytes_per_img']] = X_train.numpy().astype(np.uint8)
    train_data[:, config['data_bytes_per_img']] = y_train.numpy().astype(np.uint8)

    if balanced == 1:
        filename_prefix = f'{dataset_name}_rand_bal_sub_selection_trial={trial}_'
    else:
        filename_prefix = f'{dataset_name}_rand_sub_selection_trial={trial}_'

    # Class sequence
    class_seq = range(0, len(train_set.classes))

    seq_num = 0
    subset_idxs = np.zeros(config['N_EEPROM_BUFFER'], dtype=np.uint16)
    predicted_labels = np.zeros(config['N_TOTAL'], dtype=np.uint8)
    device_data = np.zeros(shape=(config['N_TOTAL'], config['bytes_per_img']), dtype=np.uint8)
    acc = np.zeros(len(class_seq) - 1, dtype=float)

    # Keep track of the indices of the data examples on the device with regard to the full training set
    device_data_idxs = np.zeros(config['N_TOTAL'], dtype=np.uint16)

    # Start serial connection
    memory_alloc_complete = False
    with serial.Serial(config['port'], config['baudrate'], timeout=None) as ser:
        while (not memory_alloc_complete):
            line = ser.readline().decode()  # read a '\n' terminated line and convert it to string
            if line == 'Memory allocation complete\r\r\n':
                print(line, end='')
                memory_alloc_complete = True

        # Create log directory if it doesn't exist
        if not os.path.exists(config['log_dir_path']):
            os.mkdir(config['log_dir_path'])

        req_log = open(os.path.join(config['log_dir_path'], filename_prefix + 'requests_log.txt'), 'w')
        resp_log = open(os.path.join(config['log_dir_path'], filename_prefix + 'responses_log.txt'), 'w')

        # Set random seed ----------------------------------------------------------------------------------------------
        send_command(set_random_seed, seq_num=seq_num, param_list=[random_seed], ser=ser, req_log=req_log,
                     resp_log=resp_log)

        # Prime EEPROM with examples from the 0th class ----------------------------------------------------------------
        class_idxs = get_class_example_indices(train_set, class_seq[0])
        class_subset_idxs = np.random.choice(class_idxs, config['N_EEPROM_BUFFER'], replace=False)
        device_data_idxs[config['N_RAM_BUFFER'] : config['N_TOTAL']] = class_subset_idxs

        for i in range(config['N_RAM_BUFFER'], config['N_TOTAL']):
            data_example = train_data[class_subset_idxs[i - config['N_RAM_BUFFER']]]
            device_data[i] = data_example

            send_command(write_eeprom, seq_num=seq_num, param_list=[(i - config['N_RAM_BUFFER']), config['num_per_line']], ser=ser,
                         req_log=req_log, resp_log=resp_log, data_in=data_example)
            seq_num += 1

        # Write examples for next class in the sequence to RAM buffer, perform subset selection and --------------------
        # repeat for new classes
        for t in range(1, len(class_seq)):
            class_idxs = get_class_example_indices(train_set, class_seq[t])
            class_subset_idxs = np.random.choice(class_idxs, config['N_RAM_BUFFER'], replace=False)
            device_data_idxs[0:config['N_RAM_BUFFER']] = class_subset_idxs

            for i in range(config['N_RAM_BUFFER']):
                data_example = train_data[class_subset_idxs[i]]
                device_data[i] = data_example

                send_command(write_ram_buffer, seq_num=seq_num, param_list=[i, config['num_per_line']], ser=ser,
                             req_log=req_log, resp_log=resp_log, data_in=data_example)
                seq_num += 1

            # Compute dist matrix
            send_command(compute_dist_matrix, seq_num=seq_num, param_list=[], ser=ser, req_log=req_log,
                         resp_log=resp_log)
            seq_num += 1

            # Run subset selection
            send_command(rand_subset_selection, seq_num=seq_num, param_list=[balanced, 200], ser=ser, req_log=req_log,
                         resp_log=resp_log, data_out=[subset_idxs, predicted_labels])
            seq_num += 1

            num_correct = np.sum(device_data[:, config['data_bytes_per_img']] == predicted_labels)
            acc[t - 1] = num_correct / config['N_TOTAL']

            subset_data = device_data[subset_idxs]
            device_data[0:config['N_RAM_BUFFER']] = np.zeros(shape=(config['N_RAM_BUFFER'], config['bytes_per_img']))
            device_data[config['N_RAM_BUFFER']:config['N_TOTAL']] = subset_data

            subset_data_idxs = device_data_idxs[subset_idxs]
            device_data_idxs[config['N_RAM_BUFFER']:config['N_TOTAL']] = subset_data_idxs

        # Create results directory if it doesn't exist
        if not os.path.exists(config['results_dir_path']):
            os.mkdir(config['results_dir_path'])

        np.save(os.path.join(config['results_dir_path'], filename_prefix + 'acc'), acc)

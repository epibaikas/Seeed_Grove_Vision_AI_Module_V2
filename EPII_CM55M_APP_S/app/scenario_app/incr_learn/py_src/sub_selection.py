import argparse
import serial
import sys
import math
from tqdm import tqdm
import pickle
import subprocess

from protocol_functions import *
from argparse_utils import *
from util_functions import *
from data_utils import load_dataset, ACC, get_class_example_indices
from classifiers.k_nearest_neighbors_numpy import kNearestNeighbors

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser('Script for running class-incremental learning via random data subset selection '
                                     'experiments on Seeed Grove Vision AI Module V2')

    parser.add_argument('dataset', type=str, help='The name of the dataset to be used')
    parser.add_argument('sub_sel_func', type=int, help='Subset selection function (\'0\' for random '
                                                       'selection, \'1\' for random balanced selection, \'2\' for '
                                                       'random greedy, \'3\' for evolutionary')
    parser.add_argument('seq', type=str,
                        help='Enter \'high\' or  \'low\' for high\low accuracy sequence of classes respectively')
    parser.add_argument('ram_buf_size', type=positive_int,
                        help='The size of RAM buffer given in KBs')
    parser.add_argument('eeprom_buf_size', type=positive_int,
                        help='The size of the EEPROM buffer given in KBs')
    parser.add_argument('trial', type=positive_int,
                        help='The experiment trial number used to adjust random seed for random sampling functions')

    args = vars(parser.parse_args())

    # Get the arguments
    dataset_name = args['dataset']
    sub_sel_func = args['sub_sel_func']
    seq_type = args['seq']
    ram_buf_size = args['ram_buf_size']
    eeprom_buf_size = args['eeprom_buf_size']
    trial = args['trial']

    # Check that the class sequence is valid
    if seq_type != 'high' and seq_type != 'low':
        raise argparse.ArgumentTypeError('Invalid sequence type')

    # Get configuration parameters
    config_dir_path = 'config/'
    config = read_config(config_dir_path, 'config_global.ini')
    config |= read_config(config_dir_path, 'config_exper.ini')

    # Adjust the random seed based on the trial number
    random_seed = config['random_seed'] + trial
    np.random.seed(random_seed)

    print(f'{dataset_name}, sub_sel_func={sub_sel_func}, seq_type={seq_type}, trial={trial}')

    # Load dataset
    device = 'cpu'
    train_set, test_set, X_train, y_train, X_test, y_test = load_dataset(dataset_name, config['datasets_dir_path'], device)
    config['bytes_per_example'] = X_train.shape[1] + 1
    config['data_bytes_per_example'] = X_train.shape[1]
    num_of_classes = len(train_set.classes)

    X_train = X_train.numpy().astype(np.uint8)
    y_train = y_train.numpy().astype(np.uint8)

    X_test = X_test.numpy().astype(np.uint8)
    y_test = y_test.numpy().astype(np.uint8)

    # Compute the absolute number of data examples that can fit in RAM and EEPROM buffers
    config['N_RAM_BUFFER'] = math.floor(ram_buf_size * 1024 / (X_train.shape[1] + 1))
    config['N_EEPROM_BUFFER'] = math.floor(eeprom_buf_size * 1024 / (X_train.shape[1] + 1))
    config['N_TOTAL'] = config['N_RAM_BUFFER'] + config['N_EEPROM_BUFFER']

    # Create kNN classifier for evaluation
    classifier = kNearestNeighbors(X_train, y_train)

    if not os.path.exists(config['artifacts_dir_path']):
        os.mkdir(config['artifacts_dir_path'])

    path_1 = os.path.join(config['artifacts_dir_path'], dataset_name + '_dists.npy')
    path_2 = os.path.join(config['artifacts_dir_path'], dataset_name + '_sorting_idxs.npy')
    if os.path.isfile(path_1) and os.path.isfile(path_2):
        print('Precomputed distance matrix and sorting indices loaded from file.')
        classifier.dists = np.load(path_1)
        classifier.sorting_idxs = np.load(path_2)
    else:
        print('Computing distance matrix and sorting indices...')
        classifier.train(X_test, bitshift=12)
        np.save(path_1, classifier.dists)
        np.save(path_2, classifier.sorting_idxs)

    train_data = np.zeros(shape=(len(train_set), config['bytes_per_example']), dtype=np.uint8)
    train_data[:, 0:config['data_bytes_per_example']] = X_train.astype(np.uint8)
    train_data[:, config['data_bytes_per_example']] = y_train.astype(np.uint8)

    exp_param = (f'sub_selection_emulation={str(config["host"]).lower()}_seq={seq_type}_ram_buf_size={ram_buf_size}_eeprom_buf_size='
                 f'{eeprom_buf_size}_')
    if sub_sel_func == 0:
        filename_prefix = f'{dataset_name}_rand_' + exp_param + f'trial={trial}_'
        sel_func = rand_subset_selection
        sel_func_param = [0, 200]
    elif sub_sel_func == 1:
        filename_prefix = f'{dataset_name}_rand_bal_' + exp_param + f'trial={trial}_'
        sel_func = rand_subset_selection
        sel_func_param = [1, 200]
    elif sub_sel_func == 2:
        filename_prefix = f'{dataset_name}_rand_greedy_' + exp_param + f'num_iter={config["num_iter"]}_trial={trial}_'
        sel_func = rand_greedy_subset_selection
        sel_func_param = [config['num_iter'], 200]
    elif sub_sel_func == 3:
        filename_prefix = f'{dataset_name}_evo_' + exp_param + f'num_gen={config["num_gen"]}_trial={trial}_'
        sel_func = evo_subset_selection
        sel_func_param = [config['num_gen'], 200]


    # Class sequence
    class_sequences = np.load(os.path.join(config['artifacts_dir_path'], dataset_name + '_class_sequences.npy'))
    class_seq = class_sequences[0] if seq_type == 'high' else class_sequences[1]

    test_set_1 = get_class_example_indices(test_set, class_seq[0])
    test_set_1 += get_class_example_indices(test_set, class_seq[1])

    test_sets = []
    test_sets.append(test_set_1)
    test_sets += [get_class_example_indices(test_set, class_num) for class_num in class_seq[2:]]

    train_sets = []

    seq_num = 0
    subset_idxs = np.zeros(config['N_EEPROM_BUFFER'], dtype=np.uint16)
    optim_func_buffer = np.zeros(sel_func_param[0], dtype=float)

    # Keep track of the data examples that are currently on the device
    device_data = np.zeros(shape=(config['N_TOTAL'], config['bytes_per_example']), dtype=np.uint8)

    # Keep track of the indices of the data examples on the device with regard to the full training set
    device_data_idxs = np.zeros(config['N_TOTAL'], dtype=np.uint32)

    acc_matrix = np.zeros(shape=(len(class_seq) - 1, len(class_seq) - 1), dtype=float)
    acc_test_set_union = np.zeros(shape=(len(class_seq) - 1), dtype=float)
    acc_train_set_union = np.zeros(shape=(len(class_seq) - 1), dtype=float)
    acc_global = np.zeros(shape=(len(class_seq) - 1), dtype=float)

    # Store the indices of the examples placed in EEPROM referenced with regard to the full training set
    EEPROM_trainset_idxs = []

    # Create log/txt directory if it doesn't exist
    log_txt_dir_path = os.path.join(config['log_dir_path'], 'txt')
    if not os.path.exists(log_txt_dir_path):
        os.makedirs(log_txt_dir_path)

    req_log_txt_file_path = os.path.join(log_txt_dir_path, filename_prefix + 'requests_log.txt')
    resp_log_txt_file_path = os.path.join(log_txt_dir_path, filename_prefix + 'responses_log.txt')
    req_logger, resp_logger = get_loggers(req_log_txt_file_path, resp_log_txt_file_path, debug=config['debug'])

    # Create log/xml directory if it doesn't exist
    log_xml_dir_path = os.path.join(config['log_dir_path'], 'xml')
    if not os.path.exists(log_xml_dir_path):
        os.makedirs(log_xml_dir_path)

    req_log_xml_file_path = os.path.join(log_xml_dir_path, filename_prefix + 'requests_log.xml')
    resp_log_xml_file_path = os.path.join(log_xml_dir_path, filename_prefix + 'responses_log.xml')

    # Create root elements
    req_log_xml_root = ET.Element('requests')
    resp_log_xml_root = ET.Element('response_log')

    util = {'req_logger': req_logger,
            'resp_logger': resp_logger,
            'req_log_xml_root': req_log_xml_root,
            'resp_log_xml_root': resp_log_xml_root,
            'debug': config['debug']}

    if config['host']:
        device_emulation = subprocess.Popen([os.path.join('./', config['build_dir_path'], config['binary_name'])],
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=False)
        util['writer'] = device_emulation.stdin
        util['reader'] = device_emulation.stdout
    else:
        # Start serial connection
        ser = serial.Serial(config['port'], config['baudrate'], timeout=None)
        board_init(ser)
        util['writer'] = ser
        util['reader'] = ser


    # Set random seed ----------------------------------------------------------------------------------------------
    send_command(set_random_seed, seq_num=seq_num, param_list=[random_seed], util=util)
    seq_num += 1

    # Set data buffer parameters -----------------------------------------------------------------------------------
    send_command(set_data_buffer_parameters, seq_num=seq_num, param_list=[config['N_RAM_BUFFER'],
                                                                          config['N_EEPROM_BUFFER'],
                                                                          config['bytes_per_example'],
                                                                          num_of_classes], util=util)
    seq_num += 1

    # Get a batch of examples from every class in the sequence, write it to RAM BUFFER, perform subset selection if ----
    # necessary and update EEPROM contents
    num_examples_in_eeprom = 0
    num_examples_total = 0
    for t in range(0, len(class_seq)):
        class_idxs = get_class_example_indices(train_set, class_seq[t])
        class_subset_idxs = np.random.choice(class_idxs, config['N_RAM_BUFFER'], replace=False)
        device_data_idxs[0:config['N_RAM_BUFFER']] = class_subset_idxs

        if t == 0:
            train_set_1 = list(class_subset_idxs)
        elif t == 1:
            train_set_1 += list(class_subset_idxs)
            train_sets.append(train_set_1)
        else:
            train_sets.append(list(class_subset_idxs))

        print(f'Writing examples from class {t+1} to RAM buffer...')
        for i in tqdm(range(config['N_RAM_BUFFER']), file=sys.stdout):
            data_example = train_data[class_subset_idxs[i]]
            device_data[i] = data_example

            send_command(write_ram_buffer, seq_num=seq_num, param_list=[i, config['num_per_line']], util=util,
                         data_in=data_example)
            seq_num += 1

        # Attempt to move new examples from RAM buffer to EEPROM
        # If there is not enough space, use subset selection
        not_enough_space = send_command(move_new_batch_to_eeprom, seq_num=seq_num, param_list=[], util=util)
        seq_num += 1

        if not_enough_space:
            # Compute dist matrix
            print('\tComputing distance matrix...')
            send_command(compute_dist_matrix, seq_num=seq_num, param_list=[], util=util)
            seq_num += 1

            if num_examples_in_eeprom + config['N_RAM_BUFFER'] < config['N_TOTAL']:
                num_examples_total = num_examples_in_eeprom + config['N_RAM_BUFFER']
            else:
                num_examples_total = config['N_TOTAL']
            predicted_labels = np.zeros(num_examples_total, dtype=np.uint8)

            # Run subset selection
            print('\tRunning subset selection...')
            send_command(sel_func, seq_num=seq_num, param_list=sel_func_param, util=util,
                         data_out=[subset_idxs, predicted_labels, optim_func_buffer])
            seq_num += 1

            # Check that the predicted labels returned by the device match the expected ones
            expected_classifier = kNearestNeighbors(device_data[:, 0:config['data_bytes_per_example']], device_data[:, config['data_bytes_per_example']])
            expected_classifier.train(device_data[:, 0:config['data_bytes_per_example']], symmetric=True, bitshift=12)

            expected_predicted_labels = expected_classifier.predict(device_data[:, 0:config['data_bytes_per_example']],
                                        subset_idxs, train_classifier=False, k=3)
            assert np.array_equal(expected_predicted_labels[0:num_examples_total], predicted_labels)
            # for i in range(num_examples_total):
            #     print('i =', i, ',', expected_predicted_labels[i], '==', predicted_labels[i], 'is',
            #           (expected_predicted_labels[i] == predicted_labels[i]))
            #     assert expected_predicted_labels[i] == predicted_labels[i]

            # Update device_data to mirror the data in EEPROM
            subset_idxs.sort()
            subset_idxs_set = set(subset_idxs)
            EEPROM_idxs = set(range(config['N_RAM_BUFFER'], config['N_TOTAL']))
            EEPROM_idxs_to_be_replaced = list(EEPROM_idxs - EEPROM_idxs.intersection(subset_idxs_set))
            EEPROM_idxs_to_be_replaced.sort()

            RAM_idxs = [i for i in subset_idxs if i < config['N_RAM_BUFFER']]
            assert len(RAM_idxs) == len(EEPROM_idxs_to_be_replaced)

            for i, idx in enumerate(RAM_idxs):
                device_data[EEPROM_idxs_to_be_replaced[i], :] = device_data[idx, :]
                device_data_idxs[EEPROM_idxs_to_be_replaced[i]] = device_data_idxs[idx]

            num_examples_in_eeprom = config['N_EEPROM_BUFFER']
        else:
            # Update device_data to mirror the data in EEPROM
            device_data[(t+1)*config['N_RAM_BUFFER'] : (t+2)*config['N_RAM_BUFFER'], :] = device_data[0:config['N_RAM_BUFFER'], :]
            device_data_idxs[(t+1)*config['N_RAM_BUFFER'] : (t+2)*config['N_RAM_BUFFER']] = device_data_idxs[0:config['N_RAM_BUFFER']]
            num_examples_in_eeprom += config['N_RAM_BUFFER']


        if t > 0:
            EEPROM_trainset_idxs.append(list(device_data_idxs[config['N_RAM_BUFFER'] : config['N_RAM_BUFFER'] + num_examples_in_eeprom]))


        test_set_union = []
        train_set_union = []
        for i in range(t):
            # Evaluate top-1 accuracy on the test set from each stage using the current subset of examples in EEPROM
            acc_matrix[t - 1, i] = ACC(classifier, X_test, y_test, subset_idxs=EEPROM_trainset_idxs[t-1], test_subset_idxs=test_sets[i])
            test_set_union += test_sets[i]
            train_set_union += train_sets[i]

        if t > 0:
            # Evaluate top-1 accuracy over the union of all test sets from the classes available up to this stage
            acc_test_set_union[t - 1] = ACC(classifier, X_test, y_test, subset_idxs=EEPROM_trainset_idxs[t-1], test_subset_idxs=test_set_union)

            # Evaluate top-1 accuracy over the union of all train examples provided to the device up to this stage
            eval_classifier = kNearestNeighbors(X_train[EEPROM_trainset_idxs[t-1]], y_train[EEPROM_trainset_idxs[t-1]])
            eval_classifier.train(X_train[train_set_union], symmetric=False, bitshift=12)
            acc_train_set_union[t - 1] = ACC(eval_classifier, X_train[train_set_union], y_train[train_set_union], subset_idxs=[])

            # Evaluate top-1 accuracy over the complete test set, containing test examples from all classes.
            acc_global[t - 1] = ACC(classifier, X_test, y_test, subset_idxs=EEPROM_trainset_idxs[t-1])

    # Create results directory if it doesn't exist
    if not os.path.exists(config['results_dir_path']):
        os.mkdir(config['results_dir_path'])

    results_dict = {'acc_matrix': acc_matrix,
                    'acc_test_set_union': acc_test_set_union,
                    'acc_train_set_union': acc_train_set_union,
                    'acc_global': acc_global,
                    'EEPROM_trainset_idxs': EEPROM_trainset_idxs}

    with open(os.path.join(config['results_dir_path'], filename_prefix + 'results_dict.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)

    # Write xml logs to file
    write_xml_files(req_log_xml_file_path, resp_log_xml_file_path, req_log_xml_root, resp_log_xml_root)

    # Kill the spawned process emulating the device
    if config['host']:
        device_emulation.stdin.close()
        device_emulation.stdout.close()
        device_emulation.stderr.close()
        device_emulation.terminate()
        device_emulation.wait()

    print('Done!\n')

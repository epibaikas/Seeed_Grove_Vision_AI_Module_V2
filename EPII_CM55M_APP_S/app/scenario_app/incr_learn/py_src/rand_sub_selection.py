import argparse
import serial
import sys
from tqdm import tqdm
import pickle

from protocol_functions import *
from argparse_utlis import *
from util_functions import *
from data_utils import load_dataset, ACC, get_random_balanced_subset_indices, get_class_example_indices
from classifiers.k_nearest_neighbors_numpy import kNearestNeighbors

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser('Script for running class-incremental learning via random data subset selection '
                                     'experiments on Seeed Grove Vision AI Module V2')

    parser.add_argument('dataset', type=str, help='The name of the dataset to be used')
    parser.add_argument('balanced', type=int, help='Balanced subset option')
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

    X_train = X_train.numpy().astype(np.uint8)
    y_train = y_train.numpy().astype(np.uint8)

    X_test = X_test.numpy().astype(np.uint8)
    y_test = y_test.numpy().astype(np.uint8)

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

    train_data = np.zeros(shape=(len(train_set), config['bytes_per_img']), dtype=np.uint8)
    train_data[:, 0:config['data_bytes_per_img']] = X_train.astype(np.uint8)
    train_data[:, config['data_bytes_per_img']] = y_train.astype(np.uint8)

    if balanced == 1:
        filename_prefix = f'{dataset_name}_rand_bal_sub_selection_trial={trial}_'
    else:
        filename_prefix = f'{dataset_name}_rand_sub_selection_trial={trial}_'

    # Class sequence
    # class_seq = range(0, len(train_set.classes))
    class_seq = [0, 1, 2, 3, 4]

    test_set_1 = get_class_example_indices(test_set, class_seq[0])
    test_set_1 += get_class_example_indices(test_set, class_seq[1])

    test_sets = []
    test_sets.append(test_set_1)
    test_sets += [get_class_example_indices(test_set, class_num) for class_num in class_seq[2:]]

    seq_num = 0
    subset_idxs = np.zeros(config['N_EEPROM_BUFFER'], dtype=np.uint16)
    predicted_labels = np.zeros(config['N_TOTAL'], dtype=np.uint8)

    # Keep track of the data examples that are currently on the device
    device_data = np.zeros(shape=(config['N_TOTAL'], config['bytes_per_img']), dtype=np.uint8)

    # Keep track of the indices of the data examples on the device with regard to the full training set
    device_data_idxs = np.zeros(config['N_TOTAL'], dtype=np.uint16)

    acc_matrix = np.zeros(shape=(len(class_seq) - 1, len(class_seq) - 1), dtype=float)
    acc_test_set_union = np.zeros(shape=(len(class_seq) - 1), dtype=float)
    acc_global = np.zeros(shape=(len(class_seq) - 1), dtype=float)

    # Store the indices of the examples placed in EEPROM after subset selection, referenced with regard to the full
    # training set
    EEPROM_trainset_idxs = np.zeros(shape=(len(class_seq) - 1, config['N_EEPROM_BUFFER']), dtype=np.uint16)

    # Start serial connection
    with serial.Serial(config['port'], config['baudrate'], timeout=None) as ser:
        board_init(ser)

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

        util = {'ser': ser,
                'req_logger': req_logger,
                'resp_logger': resp_logger,
                'req_log_xml_root': req_log_xml_root,
                'resp_log_xml_root': resp_log_xml_root,
                'debug': config['debug']}

        # Set random seed ----------------------------------------------------------------------------------------------
        send_command(set_random_seed, seq_num=seq_num, param_list=[random_seed], util=util)
        seq_num += 1

        # Prime EEPROM with examples from the 1st class ----------------------------------------------------------------
        class_idxs = get_class_example_indices(train_set, class_seq[0])
        class_subset_idxs = np.random.choice(class_idxs, config['N_EEPROM_BUFFER'], replace=False)
        device_data_idxs[config['N_RAM_BUFFER'] : config['N_TOTAL']] = class_subset_idxs

        print('Priming EEPROM with examples from 1st class...')
        for i in tqdm(range(config['N_RAM_BUFFER'], config['N_TOTAL']), file=sys.stdout):
            data_example = train_data[class_subset_idxs[i - config['N_RAM_BUFFER']]]
            device_data[i] = data_example

            send_command(write_eeprom, seq_num=seq_num, param_list=[(i - config['N_RAM_BUFFER']),
                        config['num_per_line']], util=util, data_in=data_example)
            seq_num += 1

        # Write examples for next class in the sequence to RAM buffer, perform subset selection and --------------------
        # repeat for new classes
        for t in range(1, len(class_seq)):
            class_idxs = get_class_example_indices(train_set, class_seq[t])
            class_subset_idxs = np.random.choice(class_idxs, config['N_RAM_BUFFER'], replace=False)
            device_data_idxs[0:config['N_RAM_BUFFER']] = class_subset_idxs

            print(f'Writing examples from class {t+1}...')
            for i in tqdm(range(config['N_RAM_BUFFER']), file=sys.stdout):
                data_example = train_data[class_subset_idxs[i]]
                device_data[i] = data_example

                send_command(write_ram_buffer, seq_num=seq_num, param_list=[i, config['num_per_line']], util=util,
                             data_in=data_example)
                seq_num += 1

            # Compute dist matrix
            print('\tComputing distance matrix...')
            send_command(compute_dist_matrix, seq_num=seq_num, param_list=[], util=util)
            seq_num += 1

            # Run subset selection
            print('\tRunning subset selection...')
            send_command(rand_subset_selection, seq_num=seq_num, param_list=[balanced, 200], util=util,
                         data_out=[subset_idxs, predicted_labels])
            seq_num += 1

            # Check that the predicted labels returned by the device match the expected ones
            expected_classifier = kNearestNeighbors(device_data[:, 0:config['data_bytes_per_img']], device_data[:, config['data_bytes_per_img']])
            expected_classifier.train(device_data[:, 0:config['data_bytes_per_img']], symmetric=True, bitshift=12)

            expected_predicted_labels = expected_classifier.predict(device_data[:, 0:config['data_bytes_per_img']],
                                        subset_idxs, train_classifier=False, k=3)
            assert np.array_equal(expected_predicted_labels, predicted_labels)

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

            EEPROM_trainset_idxs[t-1, :] = device_data_idxs[config['N_RAM_BUFFER']:]

            test_set_union = []
            for i in range(t):
                # Evaluate top-1 accuracy on the test set from each stage using the current subset of examples in EEPROM
                acc_matrix[t - 1, i] = ACC(classifier, X_test, y_test, subset_idxs=EEPROM_trainset_idxs[t-1, :], test_subset_idxs=test_sets[i])
                test_set_union += test_sets[i]

            # Evaluate top-1 accuracy over the union of all test sets from the classes available up to this stage
            acc_test_set_union[t - 1] = ACC(classifier, X_test, y_test, subset_idxs=EEPROM_trainset_idxs[t-1, :], test_subset_idxs=test_set_union)

            # Evaluate top-1 accuracy over the complete test set, containing test examples from all classes.
            acc_global[t - 1] = ACC(classifier, X_test, y_test, subset_idxs=EEPROM_trainset_idxs[t-1, :])

        # Create results directory if it doesn't exist
        if not os.path.exists(config['results_dir_path']):
            os.mkdir(config['results_dir_path'])

        results_dict = {'acc_matrix': acc_matrix,
                        'acc_test_set_union': acc_test_set_union,
                        'acc_global': acc_global,
                        'EEPROM_trainset_idxs': EEPROM_trainset_idxs}

        with open(os.path.join(config['results_dir_path'], filename_prefix + 'results_dict.pkl'), 'wb') as f:
            pickle.dump(results_dict, f)

        # Write xml logs to file
        write_xml_files(req_log_xml_file_path, resp_log_xml_file_path, req_log_xml_root, resp_log_xml_root)

import serial
import numpy as np
import os

from protocol_functions import *
from util_functions import *
from data_utils import load_dataset, get_random_balanced_subset_indices
from classifiers.k_nearest_neighbors_numpy import kNearestNeighbors

config_dir_path = 'config/'
config = read_config(config_dir_path)

# Set random seed
np.random.seed(config['random_seed'])

dataset_name = 'FashionMNIST'
device = 'cpu'
train_set, test_set, X_train, y_train, X_test, y_test = load_dataset(dataset_name, device)

classes = []
example_idxs = get_random_balanced_subset_indices(train_set, classes, subset_size=config['N_TOTAL'])

img_data = np.zeros(shape=(config['N_TOTAL'], config['bytes_per_img']), dtype=np.uint8)
img_data[:, 0:config['data_bytes_per_img']] = X_train[example_idxs, :].numpy().astype(np.uint8)
img_data[:, config['data_bytes_per_img']] = y_train[example_idxs].numpy().astype(np.uint8)

# Create temporary buffers for checking correctness of read values
data_read_buffer = np.zeros(config['bytes_per_img'], np.uint8)

dist_array_size = int(config['N_TOTAL'] * (config['N_TOTAL'] + 1) / 2)
dist_array = np.zeros(dist_array_size, dtype=np.uint16)

expected_classifier = kNearestNeighbors(img_data[:, 0:config['data_bytes_per_img']], img_data[:, config['data_bytes_per_img']])
expected_classifier.train(img_data[:, 0:config['data_bytes_per_img']], symmetric=True, bitshift=12)

labels_buffer = np.zeros(config['N_TOTAL'], dtype=np.uint8)
subset_idxs = np.zeros(config['N_EEPROM_BUFFER'], dtype=np.uint16)
predicted_labels = np.zeros(config['N_TOTAL'], dtype=np.uint8)

with serial.Serial(config['port'], config['baudrate'], timeout=None) as ser:
    board_init(ser)

    # Create log/txt directory if it doesn't exist
    log_txt_dir_path = os.path.join(config['log_dir_path'], 'txt')
    if not os.path.exists(log_txt_dir_path):
        os.makedirs(log_txt_dir_path)

    req_log_txt_file_path = os.path.join(log_txt_dir_path, 'requests_log.txt')
    resp_log_txt_file_path = os.path.join(log_txt_dir_path, 'responses_log.txt')
    req_logger, resp_logger = get_loggers(req_log_txt_file_path, resp_log_txt_file_path, debug=config['debug'])

    # Create log/xml directory if it doesn't exist
    log_xml_dir_path = os.path.join(config['log_dir_path'], 'xml')
    if not os.path.exists(log_xml_dir_path):
        os.makedirs(log_xml_dir_path)

    req_log_xml_file_path = os.path.join(log_xml_dir_path, 'requests_log.xml')
    resp_log_xml_file_path = os.path.join(log_xml_dir_path, 'responses_log.xml')

    # Create root elements
    req_log_xml_root = ET.Element('requests')
    resp_log_xml_root = ET.Element('response_log')

    util = {'ser': ser,
            'req_logger': req_logger,
            'resp_logger': resp_logger,
            'req_log_xml_root': req_log_xml_root,
            'resp_log_xml_root': resp_log_xml_root,
            'debug': config['debug']}

    seq_num = 0
    command_return_value = send_command(set_random_seed, seq_num=seq_num, param_list=[config['random_seed']], util=util)
    seq_num += 1

    for i in range(config['N_TOTAL']):
        if i < config['N_RAM_BUFFER']:
            send_command(write_ram_buffer, seq_num=seq_num, param_list=[i, config['num_per_line']], util=util, data_in=img_data[i])
            seq_num += 1
            send_command(read_ram_buffer, seq_num=seq_num, param_list=[i, config['num_per_line'], config['bytes_per_img']], util=util, data_out=data_read_buffer)
            seq_num += 1
            assert np.array_equal(img_data[i], data_read_buffer)

        else:
            send_command(write_eeprom, seq_num=seq_num, param_list=[(i - config['N_RAM_BUFFER']), config['num_per_line']], util=util, data_in=img_data[i])
            seq_num += 1
            send_command(read_eeprom, seq_num=seq_num, param_list=[(i - config['N_RAM_BUFFER']), config['num_per_line'], config['bytes_per_img']], util=util, data_out=data_read_buffer)
            seq_num += 1
            assert np.array_equal(img_data[i], data_read_buffer)

    send_command(compute_dist_matrix, seq_num=seq_num, param_list=[], util=util)
    seq_num += 1

    send_command(read_dist_matrix, seq_num=seq_num, param_list=[200, config['N_TOTAL']], util=util, data_out=dist_array)
    seq_num += 1

    # Check correctness of distance calculations
    for i in range(config['N_TOTAL']):
        for j in range(i, config['N_TOTAL']):
            idx = get_symmetric_2D_array_index(dist_array_size, i, j)
            # print('i={}, j={}, idx={}, {}, {}'.format(i, j, idx, expected_dist_matrix[i, j], dist_array[idx]))
            assert expected_classifier.dists[i, j] == dist_array[idx]

    send_command(read_labels_buffer, seq_num=seq_num, param_list=[200, config['N_TOTAL']], util=util, data_out=labels_buffer)
    seq_num += 1

    # Check correctness of read labels
    assert np.array_equal(img_data[:, config['bytes_per_img'] - 1], labels_buffer)

    send_command(rand_subset_selection, seq_num=seq_num, param_list=[0, 200], util=util, data_out=[subset_idxs, predicted_labels])
    seq_num += 1

    # Check if predicted labels match the expected predicted labels
    expected_predicted_labels = expected_classifier.predict(img_data[:, 0:config['data_bytes_per_img']], subset_idxs, train_classifier=False, k=3)

    for i, _ in enumerate(predicted_labels):
        print('i =', i, ',', expected_predicted_labels[i], '==', predicted_labels[i], 'is', (expected_predicted_labels[i] == predicted_labels[i]))
        assert expected_predicted_labels[i] == predicted_labels[i]

    write_xml_files(req_log_xml_file_path, resp_log_xml_file_path, req_log_xml_root, resp_log_xml_root)

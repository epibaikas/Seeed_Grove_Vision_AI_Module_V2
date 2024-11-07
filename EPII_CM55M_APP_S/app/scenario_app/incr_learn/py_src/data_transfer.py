import serial
import numpy as np
import os

from protocol_functions import *
from util_functions import *
from data_utils import load_dataset, get_class_example_indices, get_random_balanced_subset_indices

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

expected_dist_matrix = compute_distances(img_data, img_data, config['data_bytes_per_img'])

labels_buffer = np.zeros(config['N_TOTAL'], dtype=np.uint8)
subset_idxs = np.zeros(config['N_EEPROM_BUFFER'], dtype=np.uint16)
predicted_labels = np.zeros(config['N_TOTAL'], dtype=np.uint8)

memory_alloc_complete = False
with serial.Serial(config['port'], config['baudrate'], timeout=None) as ser:
    while(not memory_alloc_complete):
        line = ser.readline().decode()   # read a '\n' terminated line and convert it to string
        if line == 'Memory allocation complete\r\r\n':
            print(line, end='')
            memory_alloc_complete = True

    # Create log directory if it doesn't exist
    if not os.path.exists(config['log_dir_path']):
        os.mkdir(config['log_dir_path'])

    req_log = open(os.path.join(config['log_dir_path'], 'requests_log.txt'), 'w')
    resp_log = open(os.path.join(config['log_dir_path'], 'responses_log.txt'), 'w')

    seq_num = 0
    command_return_value = send_command(set_random_seed, seq_num=seq_num, param_list=[config['random_seed']],
                                        ser=ser, req_log=req_log, resp_log=resp_log)
    seq_num += 1

    for i in range(config['N_TOTAL']):
        if i < config['N_RAM_BUFFER']:
            send_command(write_ram_buffer, seq_num=seq_num, param_list=[i, config['num_per_line']], ser=ser, req_log=req_log, resp_log=resp_log, data_in=img_data[i])
            seq_num += 1
            send_command(read_ram_buffer, seq_num=seq_num, param_list=[i, config['num_per_line'], config['bytes_per_img']], ser=ser, req_log=req_log, resp_log=resp_log, data_out=data_read_buffer)
            seq_num += 1
            assert np.array_equal(img_data[i], data_read_buffer)

        else:
            send_command(write_eeprom, seq_num=seq_num, param_list=[(i - config['N_RAM_BUFFER']), config['num_per_line']], ser=ser, req_log=req_log, resp_log=resp_log, data_in=img_data[i])
            seq_num += 1
            send_command(read_eeprom, seq_num=seq_num, param_list=[(i - config['N_RAM_BUFFER']), config['num_per_line'], config['bytes_per_img']], ser=ser, req_log=req_log, resp_log=resp_log, data_out=data_read_buffer)
            seq_num += 1
            assert np.array_equal(img_data[i], data_read_buffer)

    send_command(compute_dist_matrix, seq_num=seq_num, param_list=[], ser=ser, req_log=req_log, resp_log=resp_log)
    seq_num += 1

    send_command(read_dist_matrix, seq_num=seq_num, param_list=[200, config['N_TOTAL']], ser=ser, req_log=req_log, resp_log=resp_log, data_out=dist_array)
    seq_num += 1

    # Check correctness of distance calculations
    for i in range(config['N_TOTAL']):
        for j in range(i, config['N_TOTAL']):
            idx = get_symmetric_2D_array_index(dist_array_size, i, j)
            # print('i={}, j={}, idx={}, {}, {}'.format(i, j, idx, expected_dist_matrix[i, j], dist_array[idx]))
            assert expected_dist_matrix[i, j] == dist_array[idx]

    send_command(read_labels_buffer, seq_num=seq_num, param_list=[200, config['N_TOTAL']], ser=ser, req_log=req_log,
                 resp_log=resp_log, data_out=labels_buffer)
    seq_num += 1

    # Check correctness of read labels
    assert np.array_equal(img_data[:, config['bytes_per_img'] - 1], labels_buffer)

    send_command(rand_subset_selection, seq_num=seq_num, param_list=[200], ser=ser, req_log=req_log,
                 resp_log=resp_log, data_out=[subset_idxs, predicted_labels])
    seq_num += 1

    # Check if predicted labels match the expected predicted labels
    expected_predicted_labels = predict_labels(img_data[:, config['data_bytes_per_img']], expected_dist_matrix, subset_idxs, k_kNN=3)

    for i, _ in enumerate(predicted_labels):
        print('i =', i, ',', expected_predicted_labels[i], '==', predicted_labels[i], 'is', (expected_predicted_labels[i] == predicted_labels[i]))
        assert expected_predicted_labels[i] == predicted_labels[i]

    req_log.close()
    resp_log.close()
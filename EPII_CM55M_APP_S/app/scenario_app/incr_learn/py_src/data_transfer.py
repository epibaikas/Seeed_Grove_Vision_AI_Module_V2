import serial
import numpy as np
import os

from protocol_functions import *
from util_functions import *
from data_utils import load_dataset, get_class_example_indices, get_random_balanced_subset_indices


port = '/dev/cu.usbmodem578D0263771'
baudrate = 921600

bytes_per_img = 785
data_bytes_per_img = bytes_per_img - 1
N_RAM_BUFFER = 400
N_EEPROM_BUFFER = 800
N_TOTAL = N_RAM_BUFFER + N_EEPROM_BUFFER
base_flash_addr = 0x00201000
num_per_line = 28

# Set random seeds
random_seed = 42
np.random.seed(random_seed)

dataset_name = ('FashionMNIST')
device = 'cpu'
train_set, test_set, X_train, y_train, X_test, y_test = load_dataset(dataset_name, device)

classes = []
example_idxs = get_random_balanced_subset_indices(train_set, classes, subset_size=N_TOTAL)

img_data = np.zeros(shape=(N_TOTAL, bytes_per_img), dtype=np.uint8)
img_data[:, 0:data_bytes_per_img] = X_train[example_idxs, :].numpy().astype(np.uint8)
img_data[:, data_bytes_per_img] = y_train[example_idxs].numpy().astype(np.uint8)

# Create temporary buffers for checking correctness of read values
data_read_buffer = np.zeros(bytes_per_img, np.uint8)

dist_array_size = int(N_TOTAL * (N_TOTAL + 1) / 2)
dist_array = np.zeros(dist_array_size, dtype=np.uint16)

expected_dist_matrix = compute_distances(img_data, img_data, data_bytes_per_img)

labels_buffer = np.zeros(N_TOTAL, dtype=np.uint8)
subset_idxs = np.zeros(N_EEPROM_BUFFER, dtype=np.uint16)
predicted_labels = np.zeros(N_TOTAL, dtype=np.uint8)

memory_alloc_complete = False
with serial.Serial(port, baudrate, timeout=None) as ser:
    while(not memory_alloc_complete):
        line = ser.readline().decode()   # read a '\n' terminated line and convert it to string
        if line == 'Memory allocation complete\r\r\n':
            print(line, end='')
            memory_alloc_complete = True

    # Create log directory if it doesn't exist
    log_dir_path = 'log/'
    if not os.path.exists(log_dir_path):
        os.mkdir(log_dir_path)

    req_log = open(os.path.join(log_dir_path, 'requests_log.txt'), 'w')
    resp_log = open(os.path.join(log_dir_path, 'responses_log.txt'), 'w')

    seq_num = 0

    for i in range(N_TOTAL):
        if i < N_RAM_BUFFER:
            send_command(write_ram_buffer, seq_num=seq_num, param_list=[i, num_per_line], ser=ser, req_log=req_log, resp_log=resp_log, data_in=img_data)
            seq_num += 1
            send_command(read_ram_buffer, seq_num=seq_num, param_list=[i, num_per_line, bytes_per_img], ser=ser, req_log=req_log, resp_log=resp_log, data_out=data_read_buffer)
            seq_num += 1
            assert np.array_equal(img_data[i], data_read_buffer)

        else:
            send_command(write_eeprom, seq_num=seq_num, param_list=[(i - N_RAM_BUFFER), num_per_line], ser=ser, req_log=req_log, resp_log=resp_log, data_in=img_data[i])
            seq_num += 1
            send_command(read_eeprom, seq_num=seq_num, param_list=[(i - N_RAM_BUFFER), num_per_line, bytes_per_img], ser=ser, req_log=req_log, resp_log=resp_log, data_out=data_read_buffer)
            seq_num += 1
            assert np.array_equal(img_data[i], data_read_buffer)

    send_command(compute_dist_matrix, seq_num=seq_num, param_list=[], ser=ser, req_log=req_log, resp_log=resp_log)
    seq_num += 1

    send_command(read_dist_matrix, seq_num=seq_num, param_list=[200, N_TOTAL], ser=ser, req_log=req_log, resp_log=resp_log, data_out=dist_array)
    seq_num += 1

    # Check correctness of distance calculations
    for i in range(N_TOTAL):
        for j in range(i, N_TOTAL):
            idx = get_symmetric_2D_array_index(dist_array_size, i, j)
            # print('i={}, j={}, idx={}, {}, {}'.format(i, j, idx, expected_dist_matrix[i, j], dist_array[idx]))
            assert expected_dist_matrix[i, j] == dist_array[idx]

    send_command(read_labels_buffer, seq_num=seq_num, param_list=[200, N_TOTAL], ser=ser, req_log=req_log,
                 resp_log=resp_log, data_out=labels_buffer)
    seq_num += 1

    # Check correctness of read labels
    assert np.array_equal(img_data[:, bytes_per_img - 1], labels_buffer)

    send_command(rand_subset_selection, seq_num=seq_num, param_list=[200], ser=ser, req_log=req_log,
                 resp_log=resp_log, data_out=[subset_idxs, predicted_labels])
    seq_num += 1

    # Check if predicted labels match the expected predicted labels
    expected_predicted_labels = predict_labels(img_data[:, data_bytes_per_img], expected_dist_matrix, subset_idxs, k_kNN=3)

    np.save(os.path.join(log_dir_path, 'subset_idxs'), subset_idxs)
    np.save(os.path.join(log_dir_path, 'expected_predicted_labels'),  expected_predicted_labels)
    np.save(os.path.join(log_dir_path, 'predicted_labels'), predicted_labels)

    for i, _ in enumerate(predicted_labels):
        print('i =', i, ',', expected_predicted_labels[i], '==', predicted_labels[i], 'is' , (expected_predicted_labels[i] == predicted_labels[i]))


    # Check if predicted labels match the expected predicted labels
    expected_predicted_labels = predict_labels(img_data[:, data_bytes_per_img], expected_dist_matrix, subset_idxs, k_kNN=3)

    req_log.close()
    resp_log.close()
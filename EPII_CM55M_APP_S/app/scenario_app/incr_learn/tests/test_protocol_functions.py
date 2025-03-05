import math

from protocol_functions import *
from util_functions import *

def increment_seq_num(seq_num):
    seq_num['value'] += 1
    return seq_num['value']

def test_set_random_seed(seq_num, config, util):
    command_return_value = send_command(set_random_seed, seq_num=seq_num['value'], param_list=[config['random_seed']], util=util)
    increment_seq_num(seq_num)
    assert command_return_value == 0

def test_set_data_buffer_parameters(seq_num, config, dataset, util):
    command_return_value = send_command(set_data_buffer_parameters, seq_num=seq_num['value'],
                                        param_list=[config['N_RAM_BUFFER'], config['N_EEPROM_BUFFER'],
                                                    config['bytes_per_example'], dataset['num_of_classes']], util=util)
    increment_seq_num(seq_num)
    assert command_return_value == 0

def test_write_read_ram_buffer(seq_num, config, device_data, util, data_read_buffer):
    for i in range(config['N_RAM_BUFFER']):
        send_command(write_ram_buffer, seq_num=seq_num['value'], param_list=[i, config['num_per_line']], util=util,
                     data_in=device_data[i])
        increment_seq_num(seq_num)
        send_command(read_ram_buffer, seq_num=seq_num['value'], param_list=[i, config['num_per_line'], config['bytes_per_example']],
                     util=util, data_out=data_read_buffer)
        increment_seq_num(seq_num)
        assert np.array_equal(device_data[i], data_read_buffer)


def test_write_read_eeprom_buffer(seq_num, config, device_data, util, data_read_buffer):
    for i in range(config['N_RAM_BUFFER'], config['N_TOTAL']):
        send_command(write_eeprom, seq_num=seq_num['value'], param_list=[(i - config['N_RAM_BUFFER']), config['num_per_line']],
                     util=util, data_in=device_data[i])
        increment_seq_num(seq_num)
        send_command(read_eeprom, seq_num=seq_num['value'], param_list=[(i - config['N_RAM_BUFFER']), config['num_per_line'], config['bytes_per_example']],
                     util=util, data_out=data_read_buffer)
        increment_seq_num(seq_num)
        assert np.array_equal(device_data[i], data_read_buffer)

    num_batches = math.floor(config['N_EEPROM_BUFFER'] / config['N_RAM_BUFFER'])

    # Update num_examples_total
    send_command(set_counters, seq_num=seq_num['value'], param_list=[(num_batches + 1) * config['N_RAM_BUFFER'], num_batches * config['N_RAM_BUFFER']], util=util)
    increment_seq_num(seq_num)

def test_move_new_batch_to_eeprom(seq_num, config, device_data, util, data_read_buffer):
    # Try to move a batch to eeprom after the eeprom has become full
    command_return_value = send_command(move_new_batch_to_eeprom, seq_num=seq_num['value'], param_list=[], util=util)
    increment_seq_num(seq_num)

    assert command_return_value is True

def test_compute_distance_matrix(seq_num, config, util, dist_array_size, dist_array, expected_classifier):
    send_command(compute_dist_matrix, seq_num=seq_num['value'], param_list=[], util=util)
    increment_seq_num(seq_num)

    send_command(read_dist_matrix, seq_num=seq_num['value'], param_list=[200, config['N_TOTAL']], util=util, data_out=dist_array)
    increment_seq_num(seq_num)

    # Check correctness of distance calculations
    for i in range(config['N_TOTAL']):
        for j in range(i, config['N_TOTAL']):
            idx = get_symmetric_2D_array_index(dist_array_size, i, j)
            assert expected_classifier.dists[i, j] == dist_array[idx]


def test_read_labels_buffer(seq_num, config, device_data, util, labels_buffer):
    send_command(read_labels_buffer, seq_num=seq_num['value'], param_list=[200, config['N_TOTAL']], util=util,
                 data_out=labels_buffer)
    increment_seq_num(seq_num)

    # Check correctness of read labels
    assert np.array_equal(device_data[:, config['bytes_per_example'] - 1], labels_buffer)


def test_rand_greedy_subset_selection(seq_num, config, device_data, util, subset_idxs, expected_classifier, data_read_buffer):
    num_batches = math.floor(config['N_EEPROM_BUFFER'] / config['N_RAM_BUFFER'])
    num_examples_total = (1 + num_batches) * config['N_RAM_BUFFER']

    predicted_labels = np.zeros(num_examples_total, dtype=np.uint8)
    optim_func_buffer = np.zeros(config['num_iter'], dtype=float)

    # Check random balanced subset selection ---------------------------------------------------------------------------
    send_command(rand_greedy_subset_selection, seq_num=seq_num['value'], param_list=[config['num_iter'], 200], util=util,
                 data_out=[subset_idxs, predicted_labels, optim_func_buffer])
    increment_seq_num(seq_num)

    # Check if predicted labels match the expected predicted labels
    expected_predicted_labels = expected_classifier.predict(device_data[0:num_examples_total, 0:config['data_bytes_per_example']], subset_idxs, train_classifier=False, k=3)
    assert np.array_equal(expected_predicted_labels[0:num_examples_total], predicted_labels)

    # Check if RAM subset data have been transferred correctly to EEPROM
    eeprom_idxs_set = set(range(config['N_RAM_BUFFER'], config['N_TOTAL']))
    subset_idxs_set = set(subset_idxs)

    eeprom_idxs_to_be_overwritten = list(eeprom_idxs_set - subset_idxs_set)
    eeprom_idxs_to_be_overwritten.sort()

    ram_subset_idxs = list(subset_idxs_set - eeprom_idxs_set)
    ram_subset_idxs.sort()

    # Check that all elements of subset_idxs are unique
    assert len(subset_idxs) == len(subset_idxs_set)
    assert len(ram_subset_idxs) == len(eeprom_idxs_to_be_overwritten)

    for i, eeprom_idx in enumerate(eeprom_idxs_to_be_overwritten):
        send_command(read_eeprom, seq_num=seq_num['value'], param_list=[(eeprom_idx - config['N_RAM_BUFFER']), config['num_per_line'], config['bytes_per_example']],
                     util=util, data_out=data_read_buffer)
        increment_seq_num(seq_num)
        assert np.array_equal(device_data[ram_subset_idxs[i]], data_read_buffer)
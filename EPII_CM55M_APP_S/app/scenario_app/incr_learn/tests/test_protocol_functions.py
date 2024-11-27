from protocol_functions import *
from util_functions import *

def increment_seq_num(seq_num):
    seq_num['value'] += 1
    return seq_num['value']

def test_set_random_seed(seq_num, config, util):
    command_return_value = send_command(set_random_seed, seq_num=seq_num['value'], param_list=[config['random_seed']], util=util)
    increment_seq_num(seq_num)
    assert command_return_value == 0


def test_write_read_ram_buffer(seq_num, config, img_data, util, data_read_buffer):
    for i in range(config['N_RAM_BUFFER']):
        send_command(write_ram_buffer, seq_num=seq_num['value'], param_list=[i, config['num_per_line']], util=util,
                     data_in=img_data[i])
        increment_seq_num(seq_num)
        send_command(read_ram_buffer, seq_num=seq_num['value'], param_list=[i, config['num_per_line'], config['bytes_per_img']],
                     util=util, data_out=data_read_buffer)
        increment_seq_num(seq_num)
        assert np.array_equal(img_data[i], data_read_buffer)


def test_write_read_eeprom_buffer(seq_num, config, img_data, util, data_read_buffer):
    for i in range(config['N_RAM_BUFFER'], config['N_TOTAL']):
        send_command(write_eeprom, seq_num=seq_num['value'], param_list=[(i - config['N_RAM_BUFFER']), config['num_per_line']],
                     util=util, data_in=img_data[i])
        increment_seq_num(seq_num)
        send_command(read_eeprom, seq_num=seq_num['value'], param_list=[(i - config['N_RAM_BUFFER']), config['num_per_line'], config['bytes_per_img']],
                     util=util, data_out=data_read_buffer)
        increment_seq_num(seq_num)
        assert np.array_equal(img_data[i], data_read_buffer)


def test_compute_distance_matrix(seq_num, config, util, dist_array_size, dist_array, expected_dist_matrix):
    send_command(compute_dist_matrix, seq_num=seq_num['value'], param_list=[], util=util)
    increment_seq_num(seq_num)

    send_command(read_dist_matrix, seq_num=seq_num['value'], param_list=[200, config['N_TOTAL']], util=util, data_out=dist_array)
    increment_seq_num(seq_num)

    # Check correctness of distance calculations
    for i in range(config['N_TOTAL']):
        for j in range(i, config['N_TOTAL']):
            idx = get_symmetric_2D_array_index(dist_array_size, i, j)
            assert expected_dist_matrix[i, j] == dist_array[idx]


def test_read_labels_buffer(seq_num, config, img_data, util, labels_buffer):
    send_command(read_labels_buffer, seq_num=seq_num['value'], param_list=[200, config['N_TOTAL']], util=util,
                 data_out=labels_buffer)
    increment_seq_num(seq_num)

    # Check correctness of read labels
    assert np.array_equal(img_data[:, config['bytes_per_img'] - 1], labels_buffer)


def test_rand_subset_selection(seq_num, config, img_data, util, subset_idxs, predicted_labels, expected_dist_matrix, data_read_buffer):

    # Check random balanced subset selection ---------------------------------------------------------------------------
    send_command(rand_subset_selection, seq_num=seq_num['value'], param_list=[1, 200], util=util,
                 data_out=[subset_idxs, predicted_labels])
    increment_seq_num(seq_num)

    # Check if predicted labels match the expected predicted labels
    expected_predicted_labels = predict_labels(img_data[:, config['data_bytes_per_img']],
                                               expected_dist_matrix, subset_idxs, k_kNN=3)

    assert np.array_equal(expected_predicted_labels, predicted_labels)

    # Check if RAM subset data have been transferred correctly to EEPROM
    eeprom_idxs_set = set(range(config['N_RAM_BUFFER'], config['N_TOTAL']))
    subset_idxs_set = set(subset_idxs)

    eeprom_idxs_to_be_overwritten = list(eeprom_idxs_set - subset_idxs_set)
    eeprom_idxs_to_be_overwritten.sort()

    ram_subset_idxs = list(subset_idxs_set - eeprom_idxs_set)
    ram_subset_idxs.sort()

    for i, eeprom_idx in enumerate(eeprom_idxs_to_be_overwritten):
        send_command(read_eeprom, seq_num=seq_num['value'], param_list=[(eeprom_idx - config['N_RAM_BUFFER']), config['num_per_line'], config['bytes_per_img']],
                     util=util, data_out=data_read_buffer)
        increment_seq_num(seq_num)
        assert np.array_equal(img_data[ram_subset_idxs[i]], data_read_buffer)
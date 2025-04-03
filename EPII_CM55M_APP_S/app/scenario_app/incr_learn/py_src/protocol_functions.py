from xml.etree import ElementTree as ET
from datetime import datetime as dt

import numpy as np

from util_functions import debug_print
def send_command(command_name, seq_num, param_list, util, data_in=None, data_out=None):
    req_xml = ET.SubElement(util['req_log_xml_root'], 'request')
    req_start_time_xml = ET.SubElement(req_xml, 'start_time')
    req_end_time_xml = ET.SubElement(req_xml, 'end_time')
    req_command_name_xml = ET.SubElement(req_xml, 'command_name')
    req_param_list_xml = ET.SubElement(req_xml, 'param_list')
    data_in_xml = ET.SubElement(req_xml, 'data_in')

    resp_xml = ET.SubElement(util['resp_log_xml_root'], 'response')
    resp_start_time_xml = ET.SubElement(resp_xml, 'start_time')
    resp_end_time_xml = ET.SubElement(resp_xml, 'end_time')
    resp_command_name_xml = ET.SubElement(resp_xml, 'command_name')
    resp_param_list_xml = ET.SubElement(resp_xml, 'param_list')
    data_out_xml = ET.SubElement(resp_xml, 'data_out')

    req_msg = 'begin {} {} '.format(seq_num, command_name.__name__) + ' '.join(
        [str(param) for param in param_list]) + '\r'
    util['req_logger'].info(req_msg)
    req_xml.set('seq_num', str(seq_num))
    req_command_name_xml.text = command_name.__name__
    req_param_list_xml.text = str(param_list)

    with np.printoptions(threshold=np.inf):
        if type(data_in) != list:
            data_in_xml.text = str([data_in])
        else:
            data_in_xml.text = str(data_in)

    util['writer'].write((req_msg + '\n').encode())
    util['writer'].flush()

    req_start_time_xml.text = str(dt.now())
    ack_line = util['reader'].readline().decode()
    debug_print(ack_line, end='', debug=util['debug'])
    util['resp_logger'].info(ack_line.rstrip())

    ack_line = ack_line.replace('\r', '').replace('\n', '')
    if ack_line != 'ack_begin {}'.format(seq_num):
        raise AssertionError('ack_begin not properly received')

    resp_xml.set('seq_num', str(seq_num))
    resp_command_name_xml.text = command_name.__name__
    resp_param_list_xml.text = str(param_list)
    resp_start_time_xml.text = str(dt.now())

    if data_in is not None:
        command_return_value = command_name(param_list, data_in, util)
    elif data_out is not None:
        command_return_value = command_name(param_list, data_out, util)
    else:
        command_return_value = command_name(param_list, util)

    req_msg = 'end {}\r'.format(seq_num)
    util['writer'].write((req_msg + '\n').encode())
    util['writer'].flush()

    util['req_logger'].info(req_msg + '\n')
    req_end_time_xml.text = str(dt.now())

    ack_line = util['reader'].readline().decode()
    debug_print(ack_line, debug=util['debug'])
    util['resp_logger'].info(ack_line)

    ack_line = ack_line.replace('\r', '').replace('\n', '')
    if ack_line != 'ack_end {}'.format(seq_num):
        raise AssertionError('ack_end not properly received')

    resp_end_time_xml.text = str(dt.now())

    with np.printoptions(threshold=np.inf):
        if type(data_out) != list:
            data_out_xml.text = str([data_out])
        else:
            data_out_xml.text = str(data_out)

    return command_return_value

def write_ram_buffer(param_list, data_example, util):
    if len(param_list) != 2:
        raise AssertionError('Incorrect param_list length')

    example_num = param_list[0]
    num_per_line = param_list[1]
    write_buffer(data_example, data_example.shape[0], num_per_line, util)
    return 0


def read_ram_buffer(param_list, data_read_buffer, util):
    if len(param_list) != 3:
        raise AssertionError('Incorrect param_list length')

    example_num = param_list[0]
    num_per_line = param_list[1]
    bytes_per_img = param_list[2]

    read_buffer(data_read_buffer, bytes_per_img, num_per_line, util)
    return 0


def write_eeprom(param_list, data_example, util):
    if len(param_list) != 2:
        raise AssertionError('Incorrect param_list length')

    example_num = param_list[0]
    num_per_line = param_list[1]
    write_buffer(data_example, data_example.shape[0], num_per_line,util)
    return 0


def read_eeprom(param_list, data_read_buffer, util):
    if len(param_list) != 3:
        raise AssertionError('Incorrect param_list length')

    example_num = param_list[0]
    num_per_line = param_list[1]
    bytes_per_img = param_list[2]

    read_buffer(data_read_buffer, bytes_per_img, num_per_line, util)
    return 0


def read_labels_buffer(param_list, labels_array, util):
    if len(param_list) != 2:
        raise AssertionError('Incorrect param_list length')

    num_per_line = param_list[0]
    size = param_list[1]

    read_buffer(labels_array, size, num_per_line, util)
    return 0


def move_new_batch_to_eeprom(param_list, util):
    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    resp_line = resp_line.replace('\r', '').replace('\n', '')
    not_enough_space = False if resp_line == 'new batch moved to eeprom' else True
    return not_enough_space

def compute_dist_matrix(param_list, util):
    if len(param_list) != 1:
        raise AssertionError('Incorrect param_list length')

    bitshift = param_list[0]

    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    resp_line = resp_line.replace('\r', '').replace('\n', '')
    if resp_line != 'done':
        raise AssertionError('resp_line not properly received')

    return 0


def read_dist_matrix(param_list, dist_array, util):
    if len(param_list) != 2:
        raise AssertionError('Incorrect param_list length')

    num_per_line = param_list[0]
    N = param_list[1]
    size = int(N * (N + 1) / 2)

    read_buffer(dist_array, size, num_per_line, util)
    return 0


def rand_subset_selection(param_list, data_out, util):
    if len(param_list) != 2:
        raise AssertionError('Incorrect param_list length')

    num_per_line = param_list[1]

    read_buffer(data_out[0], data_out[0].shape[0], num_per_line, util)

    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    resp_line = resp_line.replace('\r', '').replace('\n', '')
    if resp_line != 'subset_idxs_read_done':
        raise AssertionError('resp_line not properly received for subset_idxs read')

    read_buffer(data_out[1], data_out[1].shape[0], num_per_line, util)

    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    resp_line = resp_line.replace('\r', '').replace('\n', '')
    if resp_line != 'predicted_labels_read_done':
        raise AssertionError('resp_line not properly received for predicted_labels')

    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    resp_line = resp_line.replace('\r', '').replace('\n', '')
    resp_line_str = resp_line.split(':')[0].strip()
    time_measurements = resp_line.split(':')[1].strip()
    sub_sel_time = time_measurements.split(',')[0].strip()
    eeprom_update_time = time_measurements.split(',')[1].strip()
    if resp_line_str != 'Time measurements (us)':
        raise AssertionError('resp_line not properly received for time measurements')

    data_out[3] = [sub_sel_time, eeprom_update_time]

    return 0


def rand_greedy_subset_selection(param_list, data_out, util):
    if len(param_list) != 2:
        raise AssertionError('Incorrect param_list length')

    num_iter = param_list[0]
    num_per_line = param_list[1]

    # debugging ----------------------------------------------
    # Prints the iterations of the greedy process
    for i in range(num_iter):
        resp_line = util['reader'].readline().decode()
        debug_print(resp_line, end='', debug=util['debug'])
        util['resp_logger'].info(resp_line.rstrip())
    # --------------------------------------------------------

    read_buffer(data_out[0], data_out[0].shape[0], num_per_line, util)

    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    resp_line = resp_line.replace('\r', '').replace('\n', '')
    if resp_line != 'subset_idxs_read_done':
        raise AssertionError('resp_line not properly received for subset_idxs read')

    read_buffer(data_out[1], data_out[1].shape[0], num_per_line, util)

    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    resp_line = resp_line.replace('\r', '').replace('\n', '')
    if resp_line != 'predicted_labels_read_done':
        raise AssertionError('resp_line not properly received for predicted_labels')

    read_buffer(data_out[2], data_out[2].shape[0], num_per_line, util)

    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    resp_line = resp_line.replace('\r', '').replace('\n', '')
    if resp_line != 'optim_func_buffer_read_done':
        raise AssertionError('resp_line not properly received for optim_func_buffer')

    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    resp_line = resp_line.replace('\r', '').replace('\n', '')
    resp_line_str = resp_line.split(':')[0].strip()
    time_measurements = resp_line.split(':')[1].strip()
    sub_sel_time = time_measurements.split(',')[0].strip()
    eeprom_update_time = time_measurements.split(',')[1].strip()
    if resp_line_str != 'Time measurements (us)':
        raise AssertionError('resp_line not properly received for time measurements')

    data_out[3] = [sub_sel_time, eeprom_update_time]

    return 0

def evo_subset_selection(param_list, data_out, util):
    if len(param_list) != 2:
        raise AssertionError('Incorrect param_list length')

    num_gen = param_list[0]
    num_per_line = param_list[1]

    # debugging ----------------------------------------------
    # Prints the iterations of the greedy process
    for i in range(num_gen):
        resp_line = util['reader'].readline().decode()
        debug_print(resp_line, end='', debug=util['debug'])
        util['resp_logger'].info(resp_line.rstrip())
    # --------------------------------------------------------

    read_buffer(data_out[0], data_out[0].shape[0], num_per_line, util)

    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    resp_line = resp_line.replace('\r', '').replace('\n', '')
    if resp_line != 'subset_idxs_read_done':
        raise AssertionError('resp_line not properly received for subset_idxs read')

    read_buffer(data_out[1], data_out[1].shape[0], num_per_line, util)

    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    resp_line = resp_line.replace('\r', '').replace('\n', '')
    if resp_line != 'predicted_labels_read_done':
        raise AssertionError('resp_line not properly received for predicted_labels')

    read_buffer(data_out[2], data_out[2].shape[0], num_per_line, util)

    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    resp_line = resp_line.replace('\r', '').replace('\n', '')
    if resp_line != 'optim_func_buffer_read_done':
        raise AssertionError('resp_line not properly received for optim_func_buffer')

    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    resp_line = resp_line.replace('\r', '').replace('\n', '')
    resp_line_str = resp_line.split(':')[0].strip()
    time_measurements = resp_line.split(':')[1].strip()
    sub_sel_time = time_measurements.split(',')[0].strip()
    eeprom_update_time = time_measurements.split(',')[1].strip()
    if resp_line_str != 'Time measurements (us)':
        raise AssertionError('resp_line not properly received for time measurements')

    data_out[3] = [sub_sel_time, eeprom_update_time]

    return 0

def write_buffer(data, size, num_per_line, util):
    for i in range(0, size, num_per_line):
        if size - i >= num_per_line:
            req_msg = ' '.join(['{:3d}'.format(data[i + j]) for j in range(num_per_line)]) + ' \r'
        else:
            req_msg = ' '.join(['{:3d}'.format(data[i + j]) for j in range(size - i)]) + ' \r'

        util['writer'].write((req_msg + '\n').encode())
        util['writer'].flush()

        util['req_logger'].debug(req_msg)

        ack_line = util['reader'].readline().decode()
        debug_print(ack_line, end='', debug=util['debug'])
        util['resp_logger'].debug(ack_line.rstrip())

        ack_line = ack_line.replace('\r', '').replace('\n', '')
        req_msg = req_msg.strip('\r')
        if ack_line != req_msg:
            raise AssertionError('Echoed data do not match written data')

def read_buffer(array, size, num_per_line, util):
    for i in range(0, size, num_per_line):
        ack_line = util['reader'].readline().decode()
        debug_print(ack_line, end='', debug=util['debug'])
        util['resp_logger'].debug(ack_line.rstrip())

        values = ack_line.split()
        for j, value in enumerate(values):
            if array.dtype != 'float64':
                array[i + j] = int(value)
            else:
                array[i + j] = float(value)

        if i + num_per_line > size:
            ack_msg = 'ack {}\r'.format(size)
        else:
            ack_msg = 'ack {}\r'.format(i + num_per_line)

        util['writer'].write((ack_msg + '\n').encode())
        util['writer'].flush()

        util['req_logger'].debug(ack_msg)


def set_data_buffer_parameters(param_list, util):
    if len(param_list) != 4:
        raise AssertionError('Incorrect param_list length')

    for i in range(15):
        resp_line = util['reader'].readline().decode()
        debug_print(resp_line, end='', debug=util['debug'])
        util['resp_logger'].info(resp_line.rstrip())

    return 0

def set_random_seed(param_list, util):
    if len(param_list) != 1:
        raise AssertionError('Incorrect param_list length')

    random_seed = param_list[0]

    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    resp_line = resp_line.replace('\r', '').replace('\n', '')
    if resp_line != f'random seed set to: {random_seed}':
        raise AssertionError('resp_line not properly received')

    return 0

def set_counters(param_list, util):
    if len(param_list) != 2:
        raise AssertionError('Incorrect param_list length')

    num_examples_total = param_list[0]
    num_examples_in_eeprom = param_list[1]

    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    resp_line = resp_line.replace('\r', '').replace('\n', '')
    if resp_line != f'num_examples_total set to: {num_examples_total}':
        raise AssertionError('resp_line not properly received')

    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    resp_line = resp_line.replace('\r', '').replace('\n', '')
    if resp_line != f'num_examples_in_eeprom set to: {num_examples_in_eeprom}':
        raise AssertionError('resp_line not properly received')

    return 0

def stop(param_list, util):
    resp_line = util['reader'].readline().decode()
    debug_print(resp_line, end='', debug=util['debug'])
    util['resp_logger'].info(resp_line.rstrip())

    return 0


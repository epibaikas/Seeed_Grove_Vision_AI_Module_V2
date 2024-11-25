def send_command(command_name, seq_num, param_list, util, data_in=None, data_out=None):
    command_return_value = 0
    req_msg = 'begin {} {} '.format(seq_num, command_name.__name__) + ' '.join(
        [str(param) for param in param_list]) + '\r'
    util['req_logger'].info(req_msg)
    util['ser'].write(req_msg.encode())

    # Ensure that writing to logs has finished before moving to the next bit
    # req_log.flush()
    # resp_log.flush()

    ack_line = util['ser'].readline().decode()
    print(ack_line, end='')
    util['resp_logger'].info(ack_line.rstrip())
    if ack_line != 'ack_begin {}\r\r\n'.format(seq_num):
        raise AssertionError('ack_begin not properly received')

    if data_in is not None:
        command_return_value = command_name(param_list, data_in, util)
    elif data_out is not None:
        command_return_value = command_name(param_list, data_out, util)
    else:
        command_return_value = command_name(param_list, util)

    req_msg = 'end {}\r'.format(seq_num)
    util['ser'].write(req_msg.encode())
    util['req_logger'].info(req_msg + '\n')

    ack_line = util['ser'].readline().decode()
    print(ack_line)
    util['resp_logger'].info(ack_line)
    if ack_line != 'ack_end {}\r\r\n'.format(seq_num):
        raise AssertionError('ack_end not properly received')

    # Ensure that writing to logs has finished before moving to next command
    # req_log.flush()
    # resp_log.flush()

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


def compute_dist_matrix(param_list, util):
    resp_line = util['ser'].readline().decode()
    print(resp_line, end='')
    util['resp_logger'].info(resp_line.rstrip())

    if resp_line != 'done\r\r\n':
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

    resp_line = util['ser'].readline().decode()
    print(resp_line, end='')
    util['resp_logger'].info(resp_line.rstrip())

    if resp_line != 'subset_idxs_read_done\r\r\n':
        raise AssertionError('resp_line not properly received for subset_idxs read')

    read_buffer(data_out[1], data_out[1].shape[0], num_per_line, util)

    resp_line = util['ser'].readline().decode()
    print(resp_line, end='')
    util['resp_logger'].info(resp_line.rstrip())

    if resp_line != 'predicted_labels_read_done\r\r\n':
        raise AssertionError('resp_line not properly received for predicted_labels')

    return 0

def write_buffer(data, size, num_per_line, util):
    for i in range(0, size, num_per_line):
        if size - i >= num_per_line:
            req_msg = ' '.join(['{:3d}'.format(data[i + j]) for j in range(num_per_line)]) + ' \r'
        else:
            req_msg = ' '.join(['{:3d}'.format(data[i + j]) for j in range(size - i)]) + ' \r'

        util['ser'].write(req_msg.encode())
        util['req_logger'].debug(req_msg)

        ack_line = util['ser'].readline().decode()
        print(ack_line, end='')
        util['resp_logger'].debug(ack_line.rstrip())

        if ack_line != req_msg + '\r\n':
            raise AssertionError('Echoed data do not match written data')

def read_buffer(array, size, num_per_line, util):
    for i in range(0, size, num_per_line):
        ack_line = util['ser'].readline().decode()
        print(ack_line, end='')
        util['resp_logger'].debug(ack_line.rstrip())

        values = ack_line.split()
        for j, value in enumerate(values):
            array[i + j] = int(value)

        if i + num_per_line > size:
            ack_msg = 'ack {}\r'.format(size)
        else:
            ack_msg = 'ack {}\r'.format(i + num_per_line)

        util['ser'].write(ack_msg.encode())
        util['req_logger'].debug(ack_msg)

def set_random_seed(param_list, util):
    if len(param_list) != 1:
        raise AssertionError('Incorrect param_list length')

    random_seed = param_list[0]

    resp_line = util['ser'].readline().decode()
    print(resp_line, end='')
    util['resp_logger'].info(resp_line.rstrip())

    if resp_line != f'random seed set to: {random_seed}\r\r\n':
        raise AssertionError('resp_line not properly received')

    return 0

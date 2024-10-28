import serial
import numpy as np

def send_command(command_name, seq_num, param_list, ser, req_log, resp_log, data_in=None, data_out=None):
    req_msg = 'begin {} {} '.format(seq_num, command_name.__name__) + ' '.join(
        [str(param) for param in param_list]) + '\r'
    req_log.write(req_msg)
    ser.write(req_msg.encode())

    # Ensure that writing to logs has finished before moving to the next bit
    req_log.flush()
    resp_log.flush()

    ack_line = ser.readline().decode()
    print(ack_line, end='')
    resp_log.write(ack_line.rstrip() + '\n')
    if ack_line != 'ack_begin {}\r\r\n'.format(seq_num):
        return -1

    if data_in is not None:
        command_name(param_list, data_in, ser, req_log, resp_log)
    elif data_out is not None:
        command_name(param_list, data_out, ser, req_log, resp_log)
    else:
        command_name(param_list, ser, req_log, resp_log)

    req_msg = 'end {}\r'.format(seq_num)
    ser.write(req_msg.encode())
    req_log.write(req_msg + '\n\n')

    ack_line = ser.readline().decode()
    print(ack_line)
    resp_log.write(ack_line)
    if ack_line != 'ack_end {}\r\r\n'.format(seq_num):
        return -1

    # Ensure that writing to logs has finished before moving to next command
    req_log.flush()
    resp_log.flush()

    return 0


def write_ram_buffer(param_list, data, ser, req_log, resp_log):
    if len(param_list) != 2:
        return -1

    example_num = param_list[0]
    num_per_line = param_list[1]
    write_buffer(data[example_num], data.shape[1], num_per_line, ser, req_log, resp_log)
    return 0


def read_ram_buffer(param_list, data_read_buffer, ser, req_log, resp_log):
    if len(param_list) != 3:
        return -1

    example_num = param_list[0]
    num_per_line = param_list[1]
    bytes_per_img = param_list[2]

    read_buffer(data_read_buffer, bytes_per_img, num_per_line, ser, req_log, resp_log)
    return 0


def write_eeprom(param_list, img, ser, req_log, resp_log):
    if len(param_list) != 2:
        return -1

    flash_addr = param_list[0]
    num_per_line = param_list[1]
    write_buffer(img, img.shape[0], num_per_line, ser, req_log, resp_log)
    return 0


def read_eeprom(param_list, data_read_buffer, ser, req_log, resp_log):
    if len(param_list) != 3:
        return -1

    flash_address = param_list[0]
    num_per_line = param_list[1]
    bytes_per_img = param_list[2]

    read_buffer(data_read_buffer, bytes_per_img, num_per_line, ser, req_log, resp_log)
    return 0


def read_labels_buffer(param_list, labels_array, ser, req_log, resp_log):
    if len(param_list) != 2:
        return -1

    num_per_line = param_list[0]
    size = param_list[1]

    read_buffer(labels_array, size, num_per_line, ser, req_log, resp_log)
    return 0


def compute_dist_matrix(seq_num, ser, req_log, resp_log):
    resp_line = ser.readline().decode()
    print(resp_line, end='')
    resp_log.write(resp_line.rstrip() + '\n')

    if resp_line != 'done\r\r\n':
        return -1

    return 0


def read_dist_matrix(param_list, dist_array, ser, req_log, resp_log):
    if len(param_list) != 2:
        return -1

    num_per_line = param_list[0]
    N = param_list[1]
    size = int(N * (N + 1) / 2)

    read_buffer(dist_array, size, num_per_line, ser, req_log, resp_log)
    return 0


def rand_subset_selection(param_list, data_out, ser, req_log, resp_log):
    if len(param_list) != 1:
        return -1

    num_per_line = param_list[0]

    # TODO: Placed loop for debugging, remove after problem is fixed
    for i in range(data_out[1].shape[0]):
        resp_line = ser.readline().decode()
        print(resp_line, end='')
        resp_log.write(resp_line.rstrip() + '\n')

    read_buffer(data_out[0], data_out[0].shape[0], num_per_line, ser, req_log, resp_log)

    resp_line = ser.readline().decode()
    print(resp_line, end='')
    resp_log.write(resp_line.rstrip() + '\n')

    if resp_line != 'subset_idxs_read_done\r\r\n':
        return -1

    read_buffer(data_out[1], data_out[1].shape[0], num_per_line, ser, req_log, resp_log)

    resp_line = ser.readline().decode()
    print(resp_line, end='')
    resp_log.write(resp_line.rstrip() + '\n')

    if resp_line != 'predicted_labels_read_done\r\r\n':
        return -1

    return 0

def write_buffer(data, size, num_per_line, ser, req_log, resp_log):
    for i in range(0, size, num_per_line):
        if size - i >= num_per_line:
            req_msg = ' '.join(['{:3d}'.format(data[i + j]) for j in range(num_per_line)]) + ' \r'
        else:
            req_msg = ' '.join(['{:3d}'.format(data[i + j]) for j in range(size - i)]) + ' \r'

        ser.write(req_msg.encode())
        req_log.write(req_msg)

        ack_line = ser.readline().decode()
        print(ack_line, end='')
        resp_log.write(ack_line.rstrip() + '\n')

        if ack_line != req_msg + '\r\n':
            return -1

def read_buffer(array, size, num_per_line, ser, req_log, resp_log):
    for i in range(0, size, num_per_line):
        ack_line = ser.readline().decode()
        print(ack_line, end='')
        resp_log.write(ack_line.rstrip() + '\n')

        values = ack_line.split()
        for j, value in enumerate(values):
            array[i + j] = int(value)

        if i + num_per_line > size:
            ack_msg = 'ack {}\r'.format(size)
        else:
            ack_msg = 'ack {}\r'.format(i + num_per_line)

        ser.write(ack_msg.encode())
        req_log.write(ack_msg)
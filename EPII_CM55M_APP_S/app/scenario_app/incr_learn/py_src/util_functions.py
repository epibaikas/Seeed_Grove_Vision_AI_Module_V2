import numpy as np
import configparser
import logging
import os
from xml.etree import ElementTree as ET
from xml.dom import minidom


def get_symmetric_2D_array_index(array_size, i, j):
    if i > j:
        # Because of symmetry: A[i][j] == A[j][i], so swap i and j
        temp = i
        i = j
        j = temp

    index = int(((j * (j + 1)) / 2) + i)
    if index >= array_size:
        print('index_error: index exceeds array size')
        return -1

    return index

def compute_distances(A, B, data_bytes_per_img):
    A = A[:, 0:data_bytes_per_img].astype(np.uint32)
    B = B[:, 0:data_bytes_per_img].astype(np.uint32)

    A_dot = np.multiply(A, A).sum(axis=1).reshape((A.shape[0], 1)) * np.ones(shape=(1, B.shape[0]), dtype=np.uint32)
    B_dot = np.multiply(B, B).sum(axis=1) * np.ones(shape=(A.shape[0], 1), dtype=np.uint32)

    # Compute distance values and apply a 12-bit right shift
    dist_matrix = (A_dot + B_dot - 2 * A @ B.T) >> 12

    # Set every cell along the diagonal equal to 0xFFFF
    for i in range(A.shape[0]):
        dist_matrix[i, i] = 0xFFFF

    return dist_matrix.astype(np.uint16)

def predict_labels(y_train, dist_matrix, subset_idxs, k_kNN):
    sorting_subset_idxs = np.argsort(dist_matrix[:, subset_idxs], axis=1, kind='stable')
    y_train_subset = y_train[subset_idxs]
    kNN_labels = y_train_subset[sorting_subset_idxs[:, 0:k_kNN]]

    # Find the most common label (in case of a tie, select the smallest label)
    kNN_label_counts = [np.bincount(row) for row in kNN_labels]
    y_pred = np.stack([np.argmax(row) for row in kNN_label_counts])
    return y_pred

def read_config(config_dir_path):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(os.path.join(config_dir_path, 'config.ini'))

    # Access values from the configuration file
    # Settings
    port = config.get(section='settings', option='port')
    baudrate = config.getint(section='settings', option='baudrate')
    bytes_per_img = config.getint(section='settings', option='bytes_per_img')
    data_bytes_per_img = bytes_per_img - 1

    N_RAM_BUFFER = config.getint(section='settings', option='N_RAM_BUFFER')
    N_EEPROM_BUFFER = config.getint(section='settings', option='N_EEPROM_BUFFER')
    N_TOTAL = N_RAM_BUFFER + N_EEPROM_BUFFER

    base_flash_addr = config.get(section='settings', option='base_flash_addr')
    num_per_line = config.getint(section='settings', option='num_per_line')

    random_seed = config.getint(section='settings', option='random_seed')
    debug = config.getboolean(section='settings', option='debug')

    # Paths
    log_dir_path = config.get(section='paths', option='log_dir_path')
    results_dir_path = config.get(section='paths', option='results_dir_path')
    plots_dir_path = config.get(section='paths', option='plots_dir_path')

    # Return a dictionary with the retrieved values
    config_values = {
        'port': port,
        'baudrate': baudrate,
        'bytes_per_img': bytes_per_img,
        'data_bytes_per_img': data_bytes_per_img,
        'N_RAM_BUFFER': N_RAM_BUFFER,
        'N_EEPROM_BUFFER': N_EEPROM_BUFFER,
        'N_TOTAL': N_TOTAL,
        'base_flash_addr': base_flash_addr,
        'num_per_line': num_per_line,
        'random_seed': random_seed,
        'debug': debug,
        'log_dir_path': log_dir_path,
        'results_dir_path': results_dir_path,
        'plots_dir_path': plots_dir_path
    }

    return config_values

def get_loggers(req_log_file, resp_log_file, debug=False):
    # Set logging level and formatter
    logging_level = logging.DEBUG if debug else logging.INFO
    logger_formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s', datefmt='%Y-%m-%d:%H:%M:%S')


    # Request message logger
    req_logger = logging.getLogger("req_logger")
    req_logger.setLevel(logging_level)

    req_logger_handler = logging.FileHandler(req_log_file, mode='w')
    req_logger_handler.setFormatter(logger_formatter)

    req_logger.addHandler(req_logger_handler)

    # Response message logger
    resp_logger = logging.getLogger("resp_logger")
    resp_logger.setLevel(logging_level)

    resp_logger_handler = logging.FileHandler(resp_log_file, mode='w')
    resp_logger_handler.setFormatter(logger_formatter)

    resp_logger.addHandler(resp_logger_handler)

    return req_logger, resp_logger

def write_xml_files(req_log_xml_file_path, resp_log_xml_file_path, req_log_xml_root, resp_log_xml_root):
    # Pretty-print the XML files
    req_log_xml_str = ET.tostring(req_log_xml_root, encoding="unicode")
    req_log_xml_minidom = minidom.parseString(req_log_xml_str)
    req_log_xml_str = req_log_xml_minidom.toprettyxml(indent='   ')
    req_log_xml_str = '\n'.join(xml_line for xml_line in req_log_xml_str.split('\n') if xml_line.strip())

    resp_log_xml_str = ET.tostring(resp_log_xml_root, encoding="unicode")
    resp_log_xml_minidom = minidom.parseString(resp_log_xml_str)
    resp_log_xml_str = resp_log_xml_minidom.toprettyxml(indent='   ')
    resp_log_xml_str = '\n'.join(xml_line for xml_line in resp_log_xml_str.split('\n') if xml_line.strip())

    with open(req_log_xml_file_path, 'w') as f:
        f.write(req_log_xml_str)

    with open(resp_log_xml_file_path, 'w') as f:
        f.write(resp_log_xml_str)

def board_init(ser):
    board_init_complete = False
    while not board_init_complete:
        line = ser.readline().decode()  # read a '\n' terminated line and convert it to string
        if line == 'Board initialisation complete\r\r\n':
            print(line, end='')
            board_init_complete = True

def debug_print(message, end='\n', debug=False):
    if debug:
        print(message, end=end)
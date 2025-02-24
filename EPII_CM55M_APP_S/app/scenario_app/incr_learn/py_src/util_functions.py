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

def parse_config_value(value):
    """Convert string values to appropriate types (int, float, bool, or keep as string)."""
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        return value  # Return as string if not a number or boolean

def read_config(config_dir_path, config_filename):
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case

    # Read the configuration file
    config.read(os.path.join(config_dir_path, config_filename))

    config_values = {}
    for section in config.sections():
        config_section = {key: parse_config_value(value) for key, value in config[section].items()}
        config_values |= config_section

    if 'N_RAM_BUFFER' in config_values.keys():
        config_values['N_TOTAL'] = config_values['N_RAM_BUFFER'] + config_values['N_EEPROM_BUFFER']

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
    print('Waiting for board initialisation...')
    while not board_init_complete:
        line = ser.readline().decode()  # read a '\n' terminated line and convert it to string
        line = line.strip("\r\n")
        if line == 'Board initialisation complete':
            print(line, end='')
            board_init_complete = True

def debug_print(message, end='\n', debug=False):
    if debug:
        print(message, end=end)
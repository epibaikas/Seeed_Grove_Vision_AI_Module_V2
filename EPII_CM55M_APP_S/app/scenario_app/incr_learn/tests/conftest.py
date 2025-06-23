import pytest
import serial
import os
import math
import subprocess
import numpy as np
from xml.etree import ElementTree as ET

from util_functions import read_config, get_loggers, write_xml_files, board_init
from data_utils import load_dataset, get_random_balanced_subset_indices
from classifiers.k_nearest_neighbors_numpy import kNearestNeighbors

@pytest.fixture(scope="session")
def seq_num():
    # Initialize the seq_num as a dictionary to allow in-place mutation
    return {'value': 0}

@pytest.fixture(scope='session')
def config():
    config_dir_path = 'config/'
    config = read_config(config_dir_path, 'config_global.ini')
    config |= read_config(config_dir_path, 'config_pytest.ini')
    return config

@pytest.fixture(scope='session')
def dataset(config):
    dataset_name = 'FashionMNIST'
    device = 'cpu'
    train_set, test_set, X_train, y_train, X_test, y_test = load_dataset(dataset_name, config['datasets_dir_path'], device)
    config['bytes_per_example'] = X_train.shape[1] + 1
    config['data_bytes_per_example'] = X_train.shape[1]

    dataset = {'train_set': train_set,
               'test_set': test_set,
               'X_train': X_train,
               'y_train': y_train,
               'X_test': X_test,
               'y_test': y_test,
               'num_of_classes': len(train_set.classes)}
    return dataset

@pytest.fixture(scope='session')
def device_data(config, dataset):
    device_data = np.zeros(shape=(config['N_TOTAL'], config['bytes_per_example']), dtype=np.uint8)

    # RAM data
    example_idxs = get_random_balanced_subset_indices(dataset['train_set'], classes=[0], subset_size=config['N_RAM_BUFFER'])
    device_data[0:config['N_RAM_BUFFER'], 0:config['data_bytes_per_example']] = dataset['X_train'][example_idxs, :].numpy().astype(np.uint8)
    device_data[0:config['N_RAM_BUFFER'], config['data_bytes_per_example']] = dataset['y_train'][example_idxs].numpy().astype(np.uint8)

    # EEPROM data
    num_batches = math.floor(config['N_EEPROM_BUFFER'] / config['N_RAM_BUFFER'])
    for batch_num in range(num_batches):
        example_idxs = get_random_balanced_subset_indices(dataset['train_set'], classes=[batch_num + 1], subset_size=config['N_RAM_BUFFER'])

        device_data[(batch_num + 1) * config['N_RAM_BUFFER'] : (batch_num + 2) * config['N_RAM_BUFFER'], 0:config['data_bytes_per_example']] = dataset['X_train'][example_idxs, :].numpy().astype(np.uint8)
        device_data[(batch_num + 1) * config['N_RAM_BUFFER'] : (batch_num + 2) * config['N_RAM_BUFFER'], config['data_bytes_per_example']] = dataset['y_train'][example_idxs].numpy().astype(np.uint8)

    return device_data

@pytest.fixture
def data_read_buffer(config):
    return np.zeros(config['bytes_per_example'], np.uint8)


@pytest.fixture
def dist_array_size(config):
    return int(config['N_TOTAL'] * (config['N_TOTAL'] + 1) / 2)

@pytest.fixture
def dist_array(dist_array_size):
    return np.zeros(dist_array_size, dtype=np.uint16)

@pytest.fixture
def expected_classifier(config, device_data):
    classifier = kNearestNeighbors(device_data[:, 0:config['data_bytes_per_example']], device_data[:, config['data_bytes_per_example']])
    classifier.train(device_data[:, 0:config['data_bytes_per_example']], symmetric=True, bitshift=config['bitshift'])
    return classifier

@pytest.fixture
def labels_buffer(config):
    return np.zeros(config['N_TOTAL'], dtype=np.uint8)

@pytest.fixture
def subset_idxs(config):
    return np.zeros(config['N_EEPROM_BUFFER'], dtype=np.uint16)


@pytest.fixture(scope='session', autouse=True)
def host_process(config, pytestconfig):
    if config['host']:
        device_emulation = subprocess.Popen([os.path.join('./', config['build_dir_path'], config['binary_name'])],
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=False)
        pytestconfig.host_process = device_emulation
        return device_emulation
    return None

@pytest.fixture(scope='session')
def util(config, host_process, request):
    log_txt_dir_path = os.path.join(config['log_dir_path'], 'txt')
    req_log_txt_file_path = os.path.join(log_txt_dir_path, 'test_requests_log.txt')
    resp_log_txt_file_path = os.path.join(log_txt_dir_path, 'test_responses_log.txt')
    req_logger, resp_logger = get_loggers(req_log_txt_file_path, resp_log_txt_file_path, debug=config['debug'])

    # Create root elements
    req_log_xml_root = ET.Element('requests')
    resp_log_xml_root = ET.Element('response_log')

    util = {'req_logger': req_logger,
            'resp_logger': resp_logger,
            'req_log_xml_root': req_log_xml_root,
            'resp_log_xml_root': resp_log_xml_root,
            'debug': config['debug']}

    if config['host']:
        util['writer'] = host_process.stdin
        util['reader'] = host_process.stdout
    else:
        # Start serial connection
        ser = serial.Serial(config['port_1'], config['baudrate'], timeout=None)
        board_init(ser)
        util['writer'] = ser
        util['reader'] = ser

    request.config.util = util
    return util

@pytest.hookimpl()
def pytest_sessionstart(session):
    config_dir_path = 'config/'
    config = read_config(config_dir_path, 'config_global.ini')
    config |= read_config(config_dir_path, 'config_pytest.ini')

    # Set random seed
    np.random.seed(config['random_seed'])

    # Create log/txt directory if it doesn't exist
    log_txt_dir_path = os.path.join(config['log_dir_path'], 'txt')
    if not os.path.exists(log_txt_dir_path):
        os.makedirs(log_txt_dir_path)

    # Create log/xml directory if it doesn't exist
    log_xml_dir_path = os.path.join(config['log_dir_path'], 'xml')
    if not os.path.exists(log_xml_dir_path):
        os.makedirs(log_xml_dir_path)


@pytest.hookimpl()
def pytest_sessionfinish(session, exitstatus):
    util = session.config.util

    config_dir_path = 'config/'
    config = read_config(config_dir_path, 'config_global.ini')
    config |= read_config(config_dir_path, 'config_pytest.ini')

    log_xml_dir_path = os.path.join(config['log_dir_path'], 'xml')
    req_log_xml_file_path = os.path.join(log_xml_dir_path, 'test_requests_log.xml')
    resp_log_xml_file_path = os.path.join(log_xml_dir_path, 'test_responses_log.xml')

    write_xml_files(req_log_xml_file_path, resp_log_xml_file_path,
                    util['req_log_xml_root'], util['resp_log_xml_root'])


    device_emulation = getattr(session.config, 'host_process', None)
    if config['host'] and device_emulation is not None:
        device_emulation.stdin.close()
        device_emulation.stdout.close()
        device_emulation.stderr.close()
        device_emulation.terminate()
        device_emulation.wait()
        print('\nDevice emulation process terminated.')

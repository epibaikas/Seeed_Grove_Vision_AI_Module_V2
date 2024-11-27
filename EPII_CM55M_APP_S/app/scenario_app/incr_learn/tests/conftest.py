import pytest
import serial
import os
import numpy as np
from xml.etree import ElementTree as ET

from util_functions import compute_distances, read_config, get_loggers, write_xml_files, board_init
from data_utils import load_dataset, get_random_balanced_subset_indices

@pytest.fixture(scope="session")
def seq_num():
    # Initialize the seq_num as a dictionary to allow in-place mutation
    return {'value': 0}

@pytest.fixture(scope='session')
def config():
    config_dir_path = 'config/'
    config = read_config(config_dir_path)
    return config

@pytest.fixture(scope='session')
def img_data(config):
    dataset_name = 'FashionMNIST'
    device = 'cpu'
    train_set, test_set, X_train, y_train, X_test, y_test = load_dataset(dataset_name, device)

    classes = []
    example_idxs = get_random_balanced_subset_indices(train_set, classes, subset_size=config['N_TOTAL'])

    img_data = np.zeros(shape=(config['N_TOTAL'], config['bytes_per_img']), dtype=np.uint8)
    img_data[:, 0:config['data_bytes_per_img']] = X_train[example_idxs, :].numpy().astype(np.uint8)
    img_data[:, config['data_bytes_per_img']] = y_train[example_idxs].numpy().astype(np.uint8)
    return img_data

@pytest.fixture
def data_read_buffer(config):
    return np.zeros(config['bytes_per_img'], np.uint8)


@pytest.fixture
def dist_array_size(config):
    return int(config['N_TOTAL'] * (config['N_TOTAL'] + 1) / 2)

@pytest.fixture
def dist_array(dist_array_size):
    return np.zeros(dist_array_size, dtype=np.uint16)

@pytest.fixture
def expected_dist_matrix(config, img_data):
    return compute_distances(img_data, img_data, config['data_bytes_per_img'])

@pytest.fixture
def labels_buffer(config):
    return np.zeros(config['N_TOTAL'], dtype=np.uint8)

@pytest.fixture
def subset_idxs(config):
    return np.zeros(config['N_EEPROM_BUFFER'], dtype=np.uint16)

@pytest.fixture
def predicted_labels(config):
    return np.zeros(config['N_TOTAL'], dtype=np.uint8)

@pytest.fixture(scope='session')
def util(config, request):
    log_txt_dir_path = os.path.join(config['log_dir_path'], 'txt')
    req_log_txt_file_path = os.path.join(log_txt_dir_path, 'test_requests_log.txt')
    resp_log_txt_file_path = os.path.join(log_txt_dir_path, 'test_responses_log.txt')
    req_logger, resp_logger = get_loggers(req_log_txt_file_path, resp_log_txt_file_path, debug=config['debug'])

    # Create root elements
    req_log_xml_root = ET.Element('requests')
    resp_log_xml_root = ET.Element('response_log')

    ser = serial.Serial(config['port'], config['baudrate'], timeout=None)
    # Wait for the initialisation stage on the board to be completed
    board_init(ser)

    util = {'ser': ser,
            'req_logger': req_logger,
            'resp_logger': resp_logger,
            'req_log_xml_root': req_log_xml_root,
            'resp_log_xml_root': resp_log_xml_root,
            'debug': config['debug']}

    request.config.util = util
    return util

@pytest.hookimpl()
def pytest_sessionstart(session):
    config_dir_path = 'config/'
    config = read_config(config_dir_path)

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
    config = read_config(config_dir_path)

    log_xml_dir_path = os.path.join(config['log_dir_path'], 'xml')
    req_log_xml_file_path = os.path.join(log_xml_dir_path, 'test_requests_log.xml')
    resp_log_xml_file_path = os.path.join(log_xml_dir_path, 'test_responses_log.xml')

    write_xml_files(req_log_xml_file_path, resp_log_xml_file_path,
                    util['req_log_xml_root'], util['resp_log_xml_root'])
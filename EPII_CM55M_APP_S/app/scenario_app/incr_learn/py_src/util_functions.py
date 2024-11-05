import numpy as np
import configparser
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

def read_config():
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read('config/config.ini')

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

    # Paths
    log_dir_path = config.get(section='paths', option='log_dir_path')

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
        'log_dir_path': log_dir_path
    }

    return config_values

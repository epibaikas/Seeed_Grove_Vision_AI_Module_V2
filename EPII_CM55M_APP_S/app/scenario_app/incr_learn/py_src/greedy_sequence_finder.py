# Python packages
import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
import pickle

# User-defined modules
from util_functions import *
from data_utils import load_dataset, ACC, get_class_example_indices
from classifiers.k_nearest_neighbors_numpy import kNearestNeighbors


def get_train_and_test_examples_of_class_pair(class_pair):
    M = []
    test_example_idxs = []

    for class_num in class_pair:
        M += get_class_example_indices(train_set, class_num)
        test_example_idxs += get_class_example_indices(test_set, class_num)

    return M, test_example_idxs


def find_next_class(Q, available_classes, M, test_example_idxs, acc_list, first_pair=False, high=True):
    if len(available_classes) == 0:
        return Q
    else:
        highest_acc_class = (-1, 0.0)
        lowest_acc_class = (-1, 1.0)

        for class_num in tqdm(available_classes):
            new_class_idxs = torch.nonzero((train_set.targets == class_num))
            new_class_idxs = torch.squeeze(new_class_idxs).numpy().tolist()

            new_test_example_idxs = torch.nonzero((test_set.targets == class_num))
            new_test_example_idxs = torch.squeeze(new_test_example_idxs).numpy().tolist()

            acc = ACC(classifier, X_test, y_test,
                      subset_idxs=M + new_class_idxs, test_subset_idxs=test_example_idxs + new_test_example_idxs)

            # print('Q =', Q, ', new class', class_num, ' acc = ' + f'{acc:.5f}')
            if (acc > highest_acc_class[1]):
                highest_acc_class = (class_num, acc)

            if (acc < lowest_acc_class[1]):
                lowest_acc_class = (class_num, acc)

        if (high == True):
            new_class_num = highest_acc_class[0]
            acc_list.append(highest_acc_class[1])
        if (high == False):
            new_class_num = lowest_acc_class[0]
            acc_list.append(lowest_acc_class[1])

        Q += [new_class_num]
        available_classes.remove(new_class_num)

        new_class_idxs = torch.nonzero((train_set.targets == new_class_num))
        new_class_idxs = torch.squeeze(new_class_idxs).numpy().tolist()
        M += new_class_idxs

        new_test_example_idxs = torch.nonzero((test_set.targets == new_class_num))
        new_test_example_idxs = torch.squeeze(new_test_example_idxs).numpy().tolist()
        test_example_idxs += new_test_example_idxs

        if first_pair:
            return Q
        else:
            return find_next_class(Q, available_classes, M, test_example_idxs, acc_list, first_pair=False, high=high)


def find_first_pair(available_classes):
    highest_acc_pair = ([-1, -1], 0.0)
    lowest_acc_pair = ([-1, -1], 1.0)
    fully_separable_pairs = []

    for i, class_num in enumerate(available_classes):
        remaining_classes = available_classes[i+1:]

        for second_class_num in remaining_classes:
            Q = [class_num, second_class_num]

            M, test_example_idxs = get_train_and_test_examples_of_class_pair(Q)

            acc = ACC(classifier, X_test, y_test, subset_idxs=M,
                      test_subset_idxs=test_example_idxs)

            print('Q =', Q, ' acc = ' + f'{acc:.5f}')

            if acc == 1.0:
                fully_separable_pairs.append(Q)

            if acc > highest_acc_pair[1]:
                highest_acc_pair = (Q, acc)

            if acc < lowest_acc_pair[1]:
                lowest_acc_pair = (Q, acc)

    print('Highest acc pair: ', highest_acc_pair)
    print('Lowest acc pair: ', lowest_acc_pair)
    print('fully_separable_pairs', fully_separable_pairs)
    return highest_acc_pair, lowest_acc_pair, fully_separable_pairs


# Create an argument parser to set the dataset, non-volatile memory size (in MBs) and trial number
parser = argparse.ArgumentParser(
    'Greedy algorithm for finding the class sequences with highest and lowest accuracies in a class-incremental '
    'learning scenario with non-volatile memory restrictions.')

parser.add_argument('dataset', type=str,
                    help='The name of the dataset to be used')

args = vars(parser.parse_args())

# Get the arguments
dataset_name = args['dataset']

# Get configuration parameters
config_dir_path = 'config/'
config = read_config(config_dir_path, 'config_global.ini')

# Set the device
device = 'cpu'
train_set, test_set, X_train, y_train, X_test, y_test = load_dataset(dataset_name, config['datasets_dir_path'], device)

X_train = X_train.numpy().astype(np.uint8)
y_train = y_train.numpy().astype(np.uint8)

X_test = X_test.numpy().astype(np.uint8)
y_test = y_test.numpy().astype(np.uint8)

# Initialise classifier
classifier = kNearestNeighbors(X_train, y_train)

# Load distance matrix and sorting indices if they have already been computed in the past
if not os.path.exists(config['artifacts_dir_path']):
    os.mkdir(config['artifacts_dir_path'])

path_1 = os.path.join(config['artifacts_dir_path'], dataset_name + '_dists.npy')
path_2 = os.path.join(config['artifacts_dir_path'], dataset_name + '_sorting_idxs.npy')
if os.path.isfile(path_1) and os.path.isfile(path_2):
    print('Precomputed distance matrix and sorting indices loaded from file.')
    classifier.dists = np.load(path_1)
    classifier.sorting_idxs = np.load(path_2)
else:
    print('Computing distance matrix and sorting indices...')
    classifier.train(X_test, bitshift=config['bitshift'])
    np.save(path_1, classifier.dists)
    np.save(path_2, classifier.sorting_idxs)

# Create a list of available class numbers out of which a sequence can be created
available_classes = [x for x in range(len(train_set.classes))]

# Get the class pairs with the highest and lowest accuracies
if os.path.isfile(config['artifacts_dir_path'] + dataset_name + '_sequence_first_pairs.pkl'):
    print('First sequence pairs loaded from file.')
    with open(config['artifacts_dir_path'] + dataset_name + '_sequence_first_pairs.pkl', 'rb') as handle:
        highest_acc_pair, lowest_acc_pair, fully_separable_pairs = pickle.load(handle)
else:
    print('Finding first class pairs of the sequence...')
    highest_acc_pair, lowest_acc_pair, fully_separable_pairs = find_first_pair(available_classes)
    print('Saving first sequence pairs to file...')
    with open(config['artifacts_dir_path'] + dataset_name + '_sequence_first_pairs.pkl', 'wb') as handle:
        pickle.dump((highest_acc_pair, lowest_acc_pair, fully_separable_pairs), handle, protocol=pickle.HIGHEST_PROTOCOL)

# Find the highest acc class sequence
if len(fully_separable_pairs) > 1:
    print('Testing sequences of fully separable pairs...')
    highest_acc_sum = 0.0
    highest_acc_Q = []
    highest_acc_list = []
    for class_pair in tqdm(fully_separable_pairs):
        print('Pair:', class_pair)
        M, test_example_idxs = get_train_and_test_examples_of_class_pair(class_pair)
        acc_list = [1.0]

        remaining_classes = list(set(available_classes) - set(class_pair))
        Q_high = find_next_class(class_pair, remaining_classes, M, test_example_idxs, acc_list, first_pair=True, high=True)

        # Select the sequence whose sum of accuracies is the highest
        if (sum(acc_list) > highest_acc_sum):
            highest_acc_sum = sum(acc_list)
            highest_acc_Q = Q_high
            highest_acc_list = acc_list

    M, test_example_idxs = get_train_and_test_examples_of_class_pair(highest_acc_Q[0:2])
    remaining_classes = list(set(available_classes) - set(highest_acc_Q))

    third_class_num = highest_acc_Q[2]
    new_class_idxs = torch.nonzero((train_set.targets == third_class_num))
    new_class_idxs = torch.squeeze(new_class_idxs).numpy().tolist()
    M += new_class_idxs

    new_test_example_idxs = torch.nonzero((test_set.targets == third_class_num))
    new_test_example_idxs = torch.squeeze(new_test_example_idxs).numpy().tolist()
    test_example_idxs += new_test_example_idxs

    highest_acc_Q = find_next_class(highest_acc_Q, remaining_classes, M, test_example_idxs, highest_acc_list, high=True)
else:
    # If thera aren't any fully separable pairs, use the pair with the highest acc value
    class_pair = highest_acc_pair[0]
    remaining_classes = list(set(available_classes) - set(class_pair))

    M, test_example_idxs = get_train_and_test_examples_of_class_pair(class_pair)
    highest_acc_list = [highest_acc_pair[1]]

    highest_acc_Q = find_next_class(class_pair, remaining_classes, M, test_example_idxs, highest_acc_list, high=True)

# Find the lowest acc sequence
class_pair = lowest_acc_pair[0]
remaining_classes = list(set(available_classes) - set(class_pair))
M, test_example_idxs = get_train_and_test_examples_of_class_pair(class_pair)
lowest_acc_list = [lowest_acc_pair[1]]

lowest_acc_Q = find_next_class(class_pair, remaining_classes, M, test_example_idxs, lowest_acc_list, high=False)

print('Highest accuracy sequence:', highest_acc_Q, ', acc values:', ['{:0.5f}'.format(acc) for acc in highest_acc_list])
print('Lowest accuracy sequence:', lowest_acc_Q, ', acc values:', ['{:0.5f}'.format(acc) for acc in lowest_acc_list])
print('Saving sequence to file...')
np.save(config['artifacts_dir_path'] + dataset_name + '_class_sequences.npy', np.array([highest_acc_Q, lowest_acc_Q]))
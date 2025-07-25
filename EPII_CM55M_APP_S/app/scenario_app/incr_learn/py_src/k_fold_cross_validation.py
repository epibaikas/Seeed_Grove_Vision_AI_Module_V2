# Python packages
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
import os
import argparse

# User-defined modules
from util_functions import *
from data_utils import load_dataset, ACC, get_class_example_indices
from classifiers.k_nearest_neighbors_numpy import kNearestNeighbors


if __name__ == '__main__':
    # Create an argument parser to set the dataset
    parser = argparse.ArgumentParser(
        'Script for tuning the number of Nearest Neighbors via k-fold cross validation')

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

    # Define KFold
    k_folds = 6
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    val_classifier = kNearestNeighbors(X_train, y_train)

    # Load distance matrix and sorting indices if they have already been computed in the past
    if not os.path.exists(config['artifacts_dir_path']):
        os.mkdir(config['artifacts_dir_path'])

    path_1 = os.path.join(config['artifacts_dir_path'], dataset_name + '_val_dists.npy')
    # path_2 = os.path.join(config['artifacts_dir_path'], dataset_name + '_val_sorting_idxs.npy')
    if os.path.isfile(path_1): #and os.path.isfile(path_2):
        print('Precomputed distance matrix loaded from file.')
        val_classifier.dists = np.load(path_1)
        # val_classifier.sorting_idxs = np.load(path_2)
    else:
        print('Computing distance matrix for k-fold cross validation classifier...')
        val_classifier.train(X_train, symmetric=True, bitshift=config['bitshift'])
        np.save(path_1, val_classifier.dists)
        # np.save(path_2, val_classifier.sorting_idxs)


    max_avg_acc = 0
    k_kNN = 1
    for k in [1, 3, 5, 7, 9, 11]:
        fold_acc = np.zeros(k_folds)
        avg_acc = 0
        print(f'k_kNN = {k} -----------------------------')
        for fold, (train_idxs, val_idxs) in enumerate(kfold.split(X_train, y_train)):
            fold_acc[fold] = ACC(val_classifier, X_train, y_train, subset_idxs=train_idxs, test_subset_idxs=val_idxs, k_kNN=k)

            print(f'Accuracy for fold {fold + 1}: {fold_acc[fold]:.4f}')
        avg_acc = np.mean(fold_acc)
        print(f'Avg. acc = {avg_acc:.5f}')
        print('---------------------------------------')

        if avg_acc > max_avg_acc:
            max_avg_acc = avg_acc
            k_kNN = k

    print(f'Best k_kNN = {k_kNN}')
    np.save(config['artifacts_dir_path'] + dataset_name + '_k_kNN.npy', np.array(k_kNN))

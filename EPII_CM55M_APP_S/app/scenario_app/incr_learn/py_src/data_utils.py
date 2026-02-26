import numpy as np
import os
import pickle
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
import argparse


class FeatureDataset:
    """Lightweight dataset wrapper for feature vectors extracted by 
    pre-trained neural network models. Mimics the torchvision dataset 
    interface (.data, .targets, .classes) to preserve compatibility with
    get_class_example_indices() and get_random_balanced_subset_indices().
    """
    def __init__(self, features, labels, classes):
        self.data = features                        # np.ndarray (N, D)
        self.targets = torch.tensor(labels)         # torch.Tensor (N,)
        self.classes = classes                      # list of str

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def load_dataset(dataset_name, root_dir, device):
    # Get the dataset
    if dataset_name == 'FashionMNIST':
        train_set = datasets.FashionMNIST(root=root_dir, download=True, transform=ToTensor(), train=True)
        test_set = datasets.FashionMNIST(root=root_dir, download=True, transform=ToTensor(), train=False)
    elif dataset_name == 'MNIST':
        train_set = datasets.MNIST(root=root_dir, download=True, transform=ToTensor(), train=True)
        test_set = datasets.MNIST(root=root_dir, download=True, transform=ToTensor(), train=False)
    elif dataset_name == 'EMNIST':
        train_set = datasets.EMNIST(root=root_dir, split='balanced', download=True, transform=ToTensor(), train=True)
        test_set = datasets.EMNIST(root=root_dir, split='balanced', download=True, transform=ToTensor(), train=False)
    elif dataset_name.endswith('_feat'):
        # Load pre-extracted features from a pickle file
        # Expected filename: {dataset_name}.pkl inside root_dir (artifacts dir)
        feat_path = os.path.join(root_dir, f'{dataset_name}.pkl')
        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f'Pre-extracted features not found at {feat_path}. '
                f'Run feature extraction first (see extract_features()).'
            )

        with open(feat_path, 'rb') as f:
            artifact = pickle.load(f)

        train_set = FeatureDataset(artifact['train_features'],
                                   artifact['train_labels'],
                                   artifact['classes'])
        test_set = FeatureDataset(artifact['test_features'],
                                  artifact['test_labels'],
                                  artifact['classes'])
    else:
        raise argparse.ArgumentTypeError('Unknown dataset name')

    X_train = torch.as_tensor(train_set.data, dtype=torch.int32).to(device)
    y_train = train_set.targets.type(torch.int8).to(device)

    X_test = torch.as_tensor(test_set.data, dtype=torch.int32).to(device)
    y_test = test_set.targets.type(torch.int8).to(device)

    # print('Dataset:', dataset_name)
    # print('X_train shape:', X_train.shape, X_train.dtype)
    # print('y_train shape:', y_train.shape, y_train.dtype)
    # print('X_test shape:', X_test.shape, X_test.dtype)
    # print('y_test shape:', y_test.shape, y_test.dtype, '\n')

    X_train = torch.reshape(X_train, (X_train.shape[0], -1)).to(device)
    X_test = torch.reshape(X_test, (X_test.shape[0], -1)).to(device)

    return train_set, test_set, X_train, y_train, X_test, y_test

def ACC(classifier, X_test, y_test, subset_idxs, test_subset_idxs=[], k_kNN=3):
    y_pred = classifier.predict(X_test, subset_idxs, train_classifier=False, k=k_kNN)

    if len(test_subset_idxs) == 0:
        # Consider all the test examples in y_test
        num_correct = np.sum(y_test == y_pred)
        acc = num_correct / y_test.shape[0]
    else:
        # Consider only the test examples specified by test_subset_idxs
        num_correct = np.sum(y_test[test_subset_idxs] == y_pred[test_subset_idxs])
        acc = num_correct / y_test[test_subset_idxs].shape[0]
    return acc

def get_random_balanced_subset_indices(dataset, classes, subset_size):
    idxs = []

    # dataset_size = len(dataset)
    list_of_class_idxs = []

    if not classes:
        classes = range(len(dataset.classes))

    num_of_examples_in_specified_classes = 0
    for class_num in classes:
        class_idxs = get_class_example_indices(dataset, class_num)
        num_of_examples_in_specified_classes += len(class_idxs)
        list_of_class_idxs.append(class_idxs)

    for class_idxs in list_of_class_idxs:
        class_subset_size = (int)((len(class_idxs) / num_of_examples_in_specified_classes) * subset_size)

        class_subset_idxs = np.random.choice(class_idxs, class_subset_size, replace=False)
        idxs += list(class_subset_idxs)

    return idxs
def get_class_example_indices(dataset, class_num):
    class_idxs = torch.nonzero((dataset.targets == class_num))
    class_idxs = torch.squeeze(class_idxs).numpy().tolist()
    return class_idxs

def extract_features(model, dataloader, device):
    """Run the backbone on a dataloader and return (features, labels) as numpy arrays."""
    all_features = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc='Extracting features'):
            data, labels = [_.to(device, non_blocking=True) for _ in batch]
            _ = model(data)
            all_features.append(model.proto.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)
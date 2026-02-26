import os
import pickle
import argparse

import torch
import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from util_functions import read_config
from backbone_model.model import Model
from data_utils import load_dataset, extract_features
from dataloaders.cifar100 import CIFAR100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for extracting feature vectors from datasets using pre-trained network')
    parser.add_argument('--config-net', type=str, default='pretrain_mnetv2',
                        help='Network training config file name (default: pretrain_mnetv2)')
    args = parser.parse_args()

    # Get configuration parameters
    config_dir_path = 'config/'
    config = read_config(config_dir_path, 'config_global.ini')
    config_net_training = read_config(config_dir_path, f'config_{args.config_net}.ini')
    config |= config_net_training

    device = 'cpu'
    dataset_name = f'{config["block_architecture"]}_{config["dataset"]}_feat'
    feat_path = os.path.join(config['datasets_dir_path'], f'{dataset_name}.pkl')

    if not os.path.exists(feat_path):
        # ---- Feature extraction (one-time) ----
        model = Model(config, mode='pretrain')
        model.to(device)

        checkpoint_path = os.path.join(
            config['artifacts_dir_path'],
            f'{config["block_architecture"]}_{config["dataset"]}.pth'
        )
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise RuntimeError('Trained model not found!')

        # Load all classes with shuffle=False to preserve original dataset indexing
        num_classes = config['num_classes']
        if config['dataset'] == 'cifar100':
            DatasetClass = CIFAR100
        else:
            raise ValueError(f'Unsupported dataset: {config["dataset"]}')

        trainset = DatasetClass(root=config['datasets_dir_path'], train=True,
                                download=True, index=np.arange(num_classes), base_sess=True)
        testset = DatasetClass(root=config['datasets_dir_path'], train=False,
                               download=False, index=np.arange(num_classes), base_sess=True)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=True)

        print('Extracting train features...')
        train_features, train_labels = extract_features(model, train_loader, device)
        print('Extracting test features...')
        test_features, test_labels = extract_features(model, test_loader, device)

        # Apply min-max feature scaling to [0, 255] and convert to uint8
        print('Apply min-max feature scaling...')
        feat_min = train_features.min(axis=0)
        feat_max = train_features.max(axis=0)
        scale = np.where(feat_max - feat_min > 0, feat_max - feat_min, 1.0)
        
        train_features = ((train_features - feat_min) / scale * 255).round().astype(np.uint8)
        test_features  = ((test_features  - feat_min) / scale * 255).clip(0, 255).round().astype(np.uint8)

        # Save to pickle in the format load_dataset() expects
        os.makedirs(config['datasets_dir_path'], exist_ok=True)
        artifact = {
            'train_features': train_features,
            'train_labels': train_labels,
            'test_features': test_features,
            'test_labels': test_labels,
            'classes': trainset.classes,
        }
        with open(feat_path, 'wb') as f:
            pickle.dump(artifact, f)
        print(f'Saved features to {feat_path}')

    # ---- Load extracted features via the standard interface ----
    train_set, test_set, X_train, y_train, X_test, y_test = load_dataset(
        dataset_name, config['datasets_dir_path'], device
    )

    print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')

    # ---- Visualization (PCA + t-SNE) ----
    # np.random.seed(config['random_seed'])

    # pca = PCA(n_components=50, random_state=config['random_seed'])
    # X_pca = pca.fit_transform(X_train.numpy())

    # tsne = TSNE(n_components=2, random_state=config['random_seed'], perplexity=30)
    # X_tsne = tsne.fit_transform(X_pca)

    # fig, ax = plt.subplots()
    # num_classes_to_plot = 10
    # classes = np.random.choice(np.unique(y_train.numpy()), num_classes_to_plot, replace=False)
    # tab20_cmap = plt.get_cmap('tab20')

    # for color_idx, label in enumerate(classes):
    #     indices = np.where(y_train.numpy() == label)[0]
    #     ax.scatter(X_tsne[indices, 0], X_tsne[indices, 1],
    #                color=tab20_cmap(color_idx), label=train_set.classes[label])

    # ax.legend()
    # plt.show()

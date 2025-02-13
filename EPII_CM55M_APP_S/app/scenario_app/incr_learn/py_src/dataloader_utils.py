import torch
import numpy as np

from dataloaders.cifar100 import CIFAR100

def get_base_dataloader(config, dataloader_generator):
    class_index = np.arange(config['base_class'])

    if config['dataset'] == 'cifar100':
        trainset = CIFAR100(root=config['datasets_dir_path'], train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = CIFAR100(root=config['datasets_dir_path'], train=False, download=False,
                                        index=class_index, base_sess=True)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config['batch_size'], shuffle=True,
                                              num_workers=config['num_workers'], pin_memory=True, generator=dataloader_generator)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=config['batch_size'], shuffle=False,
                                             num_workers=config['num_workers'], pin_memory=True, generator=dataloader_generator)

    return trainset, trainloader, testloader

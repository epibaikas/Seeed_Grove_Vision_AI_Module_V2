import os.path

import numpy.random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random

from torchsummary import summary
import wandb

from util_functions import *
from backbone_model.model import Model
from dataloader_utils import *

def validation(model, val_loader, criterion, device):
    val_loss_criterion = 0.0
    num_correct = 0
    num_total = 0

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            data, val_labels = [_.to(device, non_blocking=True) for _ in batch]
            output = model(data)
            loss_criterion = criterion(output, val_labels)

            val_loss_criterion += loss_criterion
            _, predicted = torch.max(output, 1)
            num_correct += (predicted == val_labels).sum().item()
            num_total += val_labels.size(0)

    val_top_1_acc = num_correct / num_total
    return val_loss_criterion / len(val_loader),  val_top_1_acc


def save_checkpoint(model, optimizer, dataloader_generator, best_val_top1_acc, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'dataloader_generator_state': dataloader_generator.get_state(),
        'best_val_top1_acc': best_val_top1_acc,
        'torch_rng_state': torch.get_rng_state(),
        'torch_cuda_rng_state': torch.cuda.get_rng_state(),
        'numpy_rng_state': numpy.random.get_state(),
        'random_state': random.getstate()
    }

    torch.save(checkpoint, path)
    print(f'Checkpoint saved at epoch {epoch + 1} with best_val_top1_acc {best_val_top1_acc:.4f}')

def load_checkpoint(model, optimizer, dataloader_generator, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    dataloader_generator.set_state(checkpoint['dataloader_generator_state'].type(torch.ByteTensor))
    start_epoch = checkpoint['epoch'] + 1
    best_val_top1_acc = checkpoint['best_val_top1_acc']
    torch.set_rng_state(checkpoint['torch_rng_state'].type(torch.ByteTensor))
    torch.cuda.set_rng_state(checkpoint['torch_cuda_rng_state'].type(torch.ByteTensor))
    np.random.set_state(checkpoint['numpy_rng_state'])
    random.setstate(checkpoint['random_state'])

    print(f'Checkpoint loaded, resuming training from end of epoch {start_epoch} with best_val_top1_acc {best_val_top1_acc:.4f}')
    return start_epoch, best_val_top1_acc


if __name__ == '__main__':
    # Get configuration parameters
    config_dir_path = 'config/'
    config = read_config(config_dir_path, 'config_global.ini')
    config_net_training = read_config(config_dir_path, 'config_pretrain_mnetv2.ini')
    config |= config_net_training

    random.seed(config['random_seed'])
    os.environ['PYTHONHASHSEED'] = str(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    torch.cuda.manual_seed(config['random_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    model = Model(config, mode='pretrain')
    # summary(model, (3, 32, 32))

    criterion = nn.CrossEntropyLoss()

    # Move model and criterion to device
    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=config['learning_rate'], nesterov=config['SGDnesterov'],
                            weight_decay=config['SGDweight_decay'], momentum=config['SGDmomentum'])

    if config['advance_augment'] == True:
        from data_augmentation.augments import Augments
        from data_augmentation.cutmix import BatchCutMixLayer
        from data_augmentation.mixup import BatchMixupLayer
        from data_augmentation.mixup import BatchMultiMixupLayer
        from data_augmentation.idty import Identity

        augs = [
            BatchMixupLayer(alpha=0.8, prob=0.4),
            BatchCutMixLayer(alpha=1.0, prob=0.4),
            Identity(prob=0.2),
        ]
        config['augments'] = Augments(augs)
    else:
        config['augments'] = None

    if not os.path.exists(config['results_dir_path']):
        os.mkdir(config['results_dir_path'])

    checkpoint_path = os.path.join(config['results_dir_path'], f'{config["block_architecture"]}_{config["dataset"]}.pth')

    dataloader_generator = torch.Generator()
    if os.path.exists(checkpoint_path):
        start_epoch, best_val_top1_acc = load_checkpoint(model, optimizer, dataloader_generator, checkpoint_path, device)
    else:
        dataloader_generator.manual_seed(config['random_seed'])
        start_epoch = 0
        best_val_top1_acc = 0.0

    trainset, train_loader, val_loader = get_base_dataloader(config, dataloader_generator)

    # Login to wandb
    os.environ["WANDB_MODE"] = 'offline'
    run = wandb.init(
        project=f'{config["block_architecture"]}_{config["dataset"]}',
        config=config_net_training
    )

    for epoch in range(start_epoch, config['max_train_iter']):
        epoch_loss_criterion = 0.0
        epoch_loss_reg = 0.0
        epoch_loss_total = 0.0
        num_correct = 0
        num_total = 0

        model.train(True)

        for i, batch in enumerate(train_loader):

            data, train_labels = [_.to(device, non_blocking=True) for _ in batch]

            # Main loss function
            optimizer.zero_grad()

            if config['augments'] is not None:
                data, lam, gt_label, gt_label_aux = config['augments'](data, train_labels)

                output = model(data)
                if gt_label_aux is not None:
                    loss1 = criterion(output, gt_label)
                    loss2 = criterion(output, gt_label_aux)
                    loss_criterion = lam * loss1 + (1 - lam) * loss2
                else:
                    loss_criterion = criterion(output, gt_label)
            else:
                output = model(data)
                loss_criterion = criterion(output, train_labels)

            # feature orthogonality loss
            if config['lambda_ortho'] != 0:
                proto = F.normalize(model.proto, dim=0, p=2)
                loss_reg = proto.t() @ proto - torch.eye(proto.shape[1], device=proto.device)
                loss_reg = torch.mean(loss_reg * loss_reg)
                loss_total = loss_criterion + config['lambda_ortho'] * loss_reg

            # Backpropagation
            loss_total.backward()
            optimizer.step()

            epoch_loss_criterion += loss_criterion
            epoch_loss_reg += loss_reg
            epoch_loss_total += loss_total.item()

            _, predicted = torch.max(output, 1)
            num_correct += (predicted == train_labels).sum().item()
            num_total += train_labels.size(0)

        val_loss_criterion, val_top1_acc = validation(model, val_loader, criterion, device)

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss_total": epoch_loss_total / len(train_loader),
            "train_loss_criterion": epoch_loss_criterion / len(train_loader),
            "train_loss_reg": epoch_loss_reg / len(train_loader),
            "train_acc@1": num_correct / num_total,
            "val_loss_criterion": val_loss_criterion,
            "val_acc@1": val_top1_acc
        })

        print(f'Epoch [{epoch + 1}/{config["max_train_iter"]}], '
              f'train_loss_total: {epoch_loss_total / len(train_loader):.4f}, '
              f'train_acc@1: {num_correct / num_total:.4f}, '
              f'val_acc@1: {val_top1_acc:.4f}')

        if val_top1_acc > best_val_top1_acc:
            best_val_top1_acc = val_top1_acc
            save_checkpoint(model, optimizer, dataloader_generator, best_val_top1_acc, epoch, checkpoint_path)
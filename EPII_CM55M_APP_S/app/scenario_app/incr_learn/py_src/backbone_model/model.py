import torch
import torch.nn as nn

from backbone_model.mobilenetv2 import MobileNetV2

class Model(nn.Module):

    def __init__(self, config, mode='pretrain'):
        super().__init__()

        self.config = config
        self.mode = mode

        if config['block_architecture'] == 'mnetv2_x4':
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 1],
                [6, 96, 3, 1],
                [6, 160, 3, 1],
                [6, 320, 1, 1],
            ]

            self.backbone = MobileNetV2(num_classes=config['dim_features'],
                                        width_mult=config['width_mult'],
                                        inverted_residual_setting=inverted_residual_setting,
                                        round_nearest=config['round_nearest'],
                                        dropout=config['dropout_rate'])

        # Pretrain Fully-Connected module
        fc_out = config['base_class']
        self.fc_pretrain = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(config['dim_features']),
            nn.Linear(config['dim_features'], fc_out, bias=False)
        )

    def forward(self, inputs):
        query_vectors = self.backbone(inputs)
        self.proto = query_vectors

        if self.mode == 'pretrain':
            output = self.fc_pretrain(query_vectors)

        return output
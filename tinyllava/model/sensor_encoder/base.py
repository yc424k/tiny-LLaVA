import os
from typing import Optional

import torch
import torch.nn as nn


class SensorEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config

    def load_model(self, **kwargs):
        pretrained_sensor_encoder_path: Optional[str] = kwargs.get('pretrained_sensor_encoder_path')
        if pretrained_sensor_encoder_path is None:
            return
        checkpoint_path = os.path.join(pretrained_sensor_encoder_path, 'pytorch_model.bin')
        if not os.path.exists(checkpoint_path):
            print(f"Sensor encoder checkpoint not found at {checkpoint_path}, initializing from scratch...")
            return
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(state_dict)
        print(f'Loading sensor encoder from {checkpoint_path}...')

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sensors, sensor_mask=None):  # pragma: no cover - interface
        raise NotImplementedError

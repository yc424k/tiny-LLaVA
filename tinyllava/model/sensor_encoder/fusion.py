import math
from typing import List, Dict, Any

import torch
import torch.nn as nn

from . import register_sensor_encoder
from .base import SensorEncoder


def _normalize_scalar(value: float, value_range: torch.Tensor) -> float:
    min_v, max_v = value_range.tolist()
    if max_v == min_v:
        return 0.0
    normalized = 2.0 * ((value - min_v) / (max_v - min_v)) - 1.0
    return float(max(-1.0, min(1.0, normalized)))


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@register_sensor_encoder('fusion')
@register_sensor_encoder('mlp')
class FusionSensorEncoder(SensorEncoder):
    """Sensor encoder with modality-wise embeddings followed by self-attention fusion."""

    NUM_TOKENS = 5

    def __init__(self, config):
        super().__init__(config)
        hidden_size = getattr(config, 'hidden_size', None)
        if hidden_size is None:
            raise ValueError('TinyLlavaConfig.hidden_size is required for FusionSensorEncoder')

        self.feature_dim = getattr(config, 'sensor_feature_dim', 256) or 256
        self.dropout_prob = getattr(config, 'sensor_dropout', 0.0)
        attention_heads = getattr(config, 'sensor_attention_heads', 8) or 8
        if self.feature_dim % attention_heads != 0:
            attention_heads = 1

        self.temp_mlp = self._build_mlp(1)
        self.hum_mlp = self._build_mlp(1)
        self.wind_mlp = self._build_mlp(2)
        self.imu_mlp = self._build_mlp(6)
        self.rel_mlp = self._build_mlp(2)

        self.positional_embedding = nn.Parameter(torch.zeros(self.NUM_TOKENS, self.feature_dim))
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=attention_heads,
            dropout=self.dropout_prob,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.feature_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.proj = nn.Linear(self.NUM_TOKENS * self.feature_dim, hidden_size)

        self.register_buffer('temp_range', torch.tensor([-20.0, 50.0]))
        self.register_buffer('hum_range', torch.tensor([0.0, 100.0]))
        self.register_buffer('acc_range', torch.tensor([-20.0, 20.0]))
        self.register_buffer('gyro_range', torch.tensor([-10.0, 10.0]))

        nn.init.normal_(self.positional_embedding, std=0.02)

    def forward(self, sensors: List[Dict[str, Any]], sensor_mask=None) -> torch.Tensor:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        token_collection = []
        for sample in sensors:
            token_collection.append(self._encode_sample(sample or {}, device, dtype))

        tokens = torch.stack(token_collection, dim=0)  # [B, 5, feature_dim]
        tokens = tokens + self.positional_embedding.unsqueeze(0).to(device=device, dtype=dtype)

        attn_output, _ = self.self_attention(tokens, tokens, tokens)
        attn_output = self.attn_norm(attn_output + tokens)
        attn_output = self.dropout(attn_output)

        flattened = attn_output.reshape(attn_output.shape[0], -1)
        projected = self.proj(flattened)
        return projected.unsqueeze(1)

    def _build_mlp(self, input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
        )

    def _encode_sample(self, sample: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        temperature = float(sample.get('temperature', 0.0))
        humidity = float(sample.get('humidity', 0.0))
        wind_direction = float(sample.get('wind_direction', 0.0))
        imu = sample.get('imu', [0.0] * 6)
        if len(imu) < 6:
            imu = list(imu) + [0.0] * (6 - len(imu))

        acc = imu[:3]
        gyro = imu[3:6]

        temp_input = torch.tensor([
            _normalize_scalar(temperature, self.temp_range)
        ], device=device, dtype=dtype)
        hum_input = torch.tensor([
            _normalize_scalar(humidity, self.hum_range)
        ], device=device, dtype=dtype)

        theta = math.radians(wind_direction)
        wind_input = torch.tensor([math.cos(theta), math.sin(theta)], device=device, dtype=dtype)

        acc_tensor = torch.tensor([
            _normalize_scalar(value, self.acc_range) for value in acc
        ], device=device, dtype=dtype)
        gyro_tensor = torch.tensor([
            _normalize_scalar(value, self.gyro_range) for value in gyro
        ], device=device, dtype=dtype)
        imu_input = torch.cat([acc_tensor, gyro_tensor], dim=0)

        gyro_x_raw, gyro_y_raw = float(gyro[0]), float(gyro[1])
        yaw = math.atan2(gyro_y_raw, gyro_x_raw + 1e-6)
        theta_rel = _wrap_angle(theta - yaw)
        rel_input = torch.tensor([math.cos(theta_rel), math.sin(theta_rel)], device=device, dtype=dtype)

        tokens = torch.stack([
            self.temp_mlp(temp_input),
            self.hum_mlp(hum_input),
            self.wind_mlp(wind_input),
            self.imu_mlp(imu_input),
            self.rel_mlp(rel_input),
        ], dim=0)
        return tokens

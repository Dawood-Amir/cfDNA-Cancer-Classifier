import numpy as np
import torch
import torch.nn as nn


class FFN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        in_dim = cfg["dataset"]["num_features"]  # 16 depending on final config
        h_dims = cfg["models"]["ffn"]["hidden_dims"]
        out_dim = cfg["dataset"]["num_classes"]
        drop = cfg["models"]["ffn"]["dropout_rate"]
        self.act_str = cfg["models"]["ffn"]["activation"]

        feature_layers = []
        prev_dim = in_dim

        for h_dim in h_dims:
            feature_layers.append(
                nn.Linear(in_features=prev_dim, out_features=h_dim, bias=True)
            )
            feature_layers.append(nn.LayerNorm(h_dim))

            # Instantiate a fresh activation instance for each layer block
            if self.act_str == "leaky_relu":
                feature_layers.append(nn.LeakyReLU())
            else:
                feature_layers.append(nn.ReLU())

            feature_layers.append(nn.Dropout(drop))
            prev_dim = h_dim

        self.feature_extractor = nn.Sequential(*feature_layers)
        self.classification_head = nn.Linear(
            in_features=prev_dim, out_features=out_dim
        )

        self.apply(self.weight_init)

    def weight_init(self, module):
        if isinstance(module, nn.Linear):
            non_linearity = (
                "leaky_relu" if self.act_str == "leaky_relu" else "relu"
            )
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity=non_linearity
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classification_head(features)
        return logits
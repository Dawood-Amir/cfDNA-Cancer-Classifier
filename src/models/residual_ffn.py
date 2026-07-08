import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + residual  # Skip connection

class ResidualFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg["dataset"]["num_features"]
        h_dim = cfg["models"]["resffn"]["hidden_dim"]  # We'll define a single size
        out_dim = cfg["dataset"]["num_classes"]
        drop = cfg["models"]["resffn"]["dropout_rate"]
        self.act_str = cfg["models"]["resffn"].get("activation", "leaky_relu")

        self.input_fc = nn.Linear(in_dim, h_dim)
        self.input_bn = nn.BatchNorm1d(h_dim)
        self.act = nn.LeakyReLU(0.01) if self.act_str == "leaky_relu" else nn.ReLU()
        self.input_drop = nn.Dropout(drop)

        # 2 Residual Blocks
        self.block1 = ResidualBlock(h_dim, drop)
        self.block2 = ResidualBlock(h_dim, drop)

        self.output = nn.Linear(h_dim, out_dim)
        self.apply(self.weight_init)

    def weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.input_bn(x)
        x = self.act(x)
        x = self.input_drop(x)
        
        x = self.block1(x)
        x = self.block2(x)
        
        return self.output(x)
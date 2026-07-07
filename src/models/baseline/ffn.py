import torch.nn as nn
import numpy as no


class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        in_dim = cfg['dataset']['num_features']#17
        h_dims = cfg['models']['ffn']['hidden_dims']#arr
        out_Dim =cfg['dataset']['num_classes']
        drop =cfg['models']['ffn']['dropout_rate']
        self.act_str = cfg['models']['ffn']['activation']

        activation_func= nn.LeakyReLU() if self.act_str == "leaky_relu" else nn.ReLU()

        feature_layers= []

        prev_dim = in_dim

        for h_dim in h_dims:
            #create hidden linear layers here 
            feature_layers.append(nn.Linear(in_features= prev_dim ,out_features= h_dim , bias=True))
            feature_layers.append(nn.LayerNorm(h_dim))
            feature_layers.append(activation_func)
            feature_layers.append(nn.Dropout(drop))
            prev_dim = h_dim

        self.feature_extractor = nn.Sequential(*feature_layers)

        self.classification_head = nn.Linear(in_features=prev_dim , out_features=out_Dim)    

        self.apply(self.weight_init)

    def weight_init(self,module):
        if isinstance(module , nn.Linear):
            non_linearity = 'leaky_relu' if self.act_str == 'leaky_relu' else 'relu'
            nn.init.kaiming_normal_(module.weight , mode='fan_out' , nonlinearity=non_linearity)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self,x):

        features = self.feature_extractor(x)
        logits = self.classification_head(features)
        return  logits


        
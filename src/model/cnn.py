import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        in_channel = cfg['dataset']['num_features']
        channels = cfg['models']['cnn']['out_channels'] # eg 32  (filters)
        out_dim =cfg['dataset']['num_classes']
        k_size =cfg['models']['cnn']['kernel_size'] # This determines how many input steps the kernel looks at when it computes one output value. For example, a kernel size of 3 means it slides over 3 input steps at a time.
        dropout =cfg['models']['cnn']['dropout_rate']

        # Storage for tracking dead neurons per epoch
        self.relu1_dead_pcts = []
        self.relu2_dead_pcts = []

        #network structure 
        self.conv_block =nn.Sequential(
            #Hidden layer1
            nn.Conv1d( in_channels =in_channel, out_channels=channels[0] ,kernel_size=k_size),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout),

            #Hidden layer2
            nn.Conv1d(in_channels= channels[0] , out_channels=channels[1] , kernel_size=k_size ),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
            )
        
        # output
        self.classifier = nn.Linear(in_features= channels[1] , out_features=out_dim)

        self.apply(self.weight_init)
        self.conv_block[2].register_forward_hook(self.relu1_hook)
        self.conv_block[6].register_forward_hook(self.relu2_hook)
 
    def weight_init(self,module):
        if isinstance(module,nn.Conv1d ):
            
            nn.init.kaiming_normal_(module.weight ,mode='fan_in' , nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
   
    def relu1_hook(self, modul, input, output:torch.Tensor ):
        if self.training:
            pct = (output == 0).float().mean().item()
            self.relu1_dead_pcts.append(pct)

    def relu2_hook(self, modul, input, output:torch.Tensor ):
        if(self.training):
            pct = (output==0).float().mean().item()
            self.relu2_dead_pcts.append(pct)

    def clear_epoch_metrics(self):
        """Resets the storage lists at the start of a new epoch."""
        self.relu1_dead_pcts = []
        self.relu2_dead_pcts = []
        
    def forward(self,x):
        x=x.unsqueeze(-1)# (B, 6, 1)
        x = self.conv_block(x)# (B, 64, 1)
        x = x.squeeze(-1) # (B, 64)
        logits = self.classifier(x)   # (B, 4)
        return logits

import torch
import torch.nn as nn
from conformer import Conformer

class conformer(nn.Module):
    def __init__(self, num_classes, input_dim, num_encoder_layers,pre_train_class, encoder_dim=512):
        super(conformer, self).__init__()
        self.encoder = Conformer(num_classes=1000, 
                  input_dim=input_dim, 
                  encoder_dim=512, 
                  num_encoder_layers=num_encoder_layers)
        
        self.classifier_en = nn.Sequential(
            nn.Linear(1000, 1000), 
        )
        self.classifier_zh = nn.Sequential(
            nn.Linear(1000, 5000), 
        )
        self.classifier_pre = nn.Sequential(
            nn.Linear(1000, pre_train_class),
           

    def forward(self, x, length):
        x, out_length = self.encoder(x,length)
        out_en = self.classifier_en(x)
        out_zh = self.classifier_zh(x)
        out_pre = self.classifier_pre(x)
        return out_en, out_zh, out_length, out_pre

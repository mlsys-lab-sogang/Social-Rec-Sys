"""
Original code: fairseq
    https://github.com/facebookresearch/fairseq/tree/main/fairseq
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DropoutModule(nn.Module):
    def __init__(self, p, module_name=None) -> None:
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False
    
    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x
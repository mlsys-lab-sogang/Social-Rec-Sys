import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class DropoutModule(nn.Module):
    """
    Original code: fairseq
        https://github.com/facebookresearch/fairseq/tree/main/fairseq
    """
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

def init_params(module, n_layers):
    """
    Initialize parameters following module type.
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal__(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
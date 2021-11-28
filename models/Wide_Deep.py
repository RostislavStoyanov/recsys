import torch
from torch import nn

from .DeepForward import DeepForward

class Wide_Deep(nn.Module):
    def __init__(self, dim_features, dim_hidden):
        super().__init__()
        
        self.dim_input = dim_features
        self.dim_output = 4 
        
        self.dim_hidden = dim_hidden
        self.dim_hidden.append(self.dim_output)
        
        self.deep = DeepForward(self.dim_input, self.dim_hidden, 0.2)
        self.wide = nn.Linear(self.dim_input, self.dim_output)
        

    def forward(self, feat_vector):
        
        out = self.deep(feat_vector)
        wide_out = self.wide(feat_vector)
        out.add_(wide_out)
        
        return out
## Implementation of Wide and deep architecture +  using a language model
import torch
from torch import nn

from transformers import XLMRobertaModel, XLMRobertaForMaskedLM

from .DeepForward import DeepForward

class Wd_Xlm_T(XLMRobertaForMaskedLM):
    def __init__(self, config, dim_features, dim_hidden):
        super().__init__(config)
        
        self.roberta = XLMRobertaModel(config, add_pooling_layer=True)
        
        self.dim_input = dim_features + 768
        self.dim_output = 4 
        
        self.dim_hidden = dim_hidden
        self.dim_hidden.append(self.dim_output)
        
        self.deep = DeepForward(self.dim_input, self.dim_hidden, 0.2)
        self.wide = nn.Linear(self.dim_input, self.dim_output)
        
        self.init_weights()

    def forward(self, input_ids, attention_mask, feat_vector):
        lm_outputs = self.roberta(input_ids, attention_mask=attention_mask)
        
        embedding = lm_outputs.pooler_output
        
        concat = torch.cat((embedding, feat_vector), 1)
        out = self.deep(concat)
        wide_out = self.wide(concat)
        out.add_(wide_out)
        
        return out
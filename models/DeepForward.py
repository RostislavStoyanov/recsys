import torch
from torch import nn

### TODO batch norm? different activations?

class DeepForward(nn.Module):

    def _create_dense_layer(self, dim_in, dim_out, p=0):
        layer = [nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True)]
        if(p>0):
            layer.append(nn.Dropout(p))

        return nn.Sequential(*layer)

 
    def __init__(self, dim_input, dim_hidden, p=None):
        '''
        dim_input: int
            dimension of input
        dim_hidden: List[int] 
            contains dimenstions of linear layers
        p: float, List[float] or None
            dropout probability, if single float same prob for each layer
            if None, no dropout is used
    
    
        '''
        super().__init__()

        if isinstance(p, float):
            self.dropout_ps = [p for _ in range(len(dim_hidden))]
        elif isinstance(p, list):
            self.dropout_ps = p
            if(len(p) < len(dim_hidden)):
                extend_list = [0.0 for _ in range(len(dim_hidden) - len(p))]
                self.dropout_ps.extend(extend_list)
        else:
            self.dropout_ps = [0.0 for _ in range(len(dim_hidden))]

        self.dim_input = dim_input
        self.dim_hidden = dim_hidden

        self.net = nn.Sequential()

        self.net.add_module(
            "dense_layer_0",
            self._create_dense_layer(self.dim_input, self.dim_hidden[0], self.dropout_ps[0])
        )

        for i in range(1, len(dim_hidden)):
            self.net.add_module(
                "dense_layer_{}".format(i),
                self._create_dense_layer(self.dim_hidden[i-1], self.dim_hidden[i], self.dropout_ps[i])
            )
    
    def forward(self, X):
        return self.net(X)
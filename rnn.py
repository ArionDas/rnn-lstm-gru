import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class RNNCell(nn.Module):
    
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        
        if self.nonlinearity not in ["tanh", "relu"]:
            raise ValueError("Invalid nonlinearity selected for RNN.")
        
        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        ## The value is inversely proportional to the square root of the hidden_size.
        ## This scaling ensures that the magnitude of weights decreases as the number of hidden units increases, 
        ## which helps avoid issues like exploding or vanishing gradients.
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std) ## Sampling from a symmetric range avoids any bias toward positive or negative values.
            
    def forward(self, input, hx=None):
        
        
        ## Inputs:
            ## input: (batch_size, input_size)
            ## hx: (batch_size, hidden_size)
        ## Outputs:
            ## hy: (batch_size, hidden_size)
        
        
        ## starting with zeroes is neutral avoiding any kind of bias  
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size)
            
        hy = (self.x2h(input) + self.h2h(hx))
        

        if self.nonlinearity == "tanh":
            hy = torch.tanh(hy)
        else:
            hy = torch.relu(hy)
            
        return hy
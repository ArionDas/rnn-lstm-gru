import torch
import torch.nn as nn
import numpy as np


class LSTMCell(nn.Module):
    
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        self.xh = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    def forward(self, input, hx=None):
        
        ## Inputs
            # input: (batch, input_size)
            # hx: (batch, hidden_size)
        
        ## Outputs
            # hx: (batch, hidden_size)
            # cy: (batch, hidden_size) 
            
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size)
            hx = (hx, hx)
            
        hx, cx = hx
        
        gates = self.xh(input) + self.hh(hx)
        
        ## gates
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
        
        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)
        
        cy = cx * f_t + i_t * g_t
        
        hy = o_t * torch.tanh(cy)
        
        return (hy, cy)
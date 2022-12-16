#%%
import torch 
import torch.nn as nn
import torch.nn.functional as F


class BlockModel(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.model(x)
    

class StaticGenerator(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, data_type):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.data_type = data_type
        
        self.block1 = BlockModel(self.input_dim, self.hidden_dim)
        self.block2 = BlockModel(self.hidden_dim, self.hidden_dim)
        self.block3 = BlockModel(self.hidden_dim, self.output_dim)
        
        self.model = nn.Sequential(
            self.block1,
            self.block2,
            self.block3
        )
        
        
    def forward(self, x):
        
        if self.data_type == 'categorical':
            return torch.round(self.model(x))
        else :       
            return self.model(x)

class StaticDiscriminator(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=None, output_dim=2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        if self.hidden_dim is None :
            self.hidden_dim = self.input_dim // 2
        
        self.block1 = BlockModel(self.input_dim, self.hidden_dim)
        self.block2 = BlockModel(self.hidden_dim, self.hidden_dim)
        self.last_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Softmax(dim=-1)
        )
        
        self.model = nn.Sequential(
            self.block1,
            self.block2,
            self.last_layer
        )
        
    def forward(self, x):
        
        return self.model(x)

#%%

    



#%%
class DynamicBlock(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, static_dim):
        super().__init__()
        '''
        performs self attention with infusioning the static data
        data : (batch_size, sequence_length, input_dim)
        static_dim : dimension_size for static_columns
        '''
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.output_dm = output_dim
        

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.multihead_attention_layer = nn.MultiheadAttention(input_dim, 8, 0.5)
        
        
        
        
    def forward(self, x, static_data):
        '''
        x : (batch_size, time_sequence)
        '''
        
        pass
        
        

#%%
class DynamicGenerator(nn.Module):
    
    def __init__(self, input_dim, output_dim, data_type):
        super().__init__()
        
        pass
    
    def forward(self, x):
        pass

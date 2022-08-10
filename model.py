import torch

class NeuralModel(torch.nn.Module):
    def __init__(self, input=4, dim=100, dropout=0.5):
        super(NeuralModel, self).__init__()
        self.linear1 = torch.nn.Linear(input, dim)
        self.act1 = torch.nn.GELU()
        self.drop1 = torch.nn.Dropout(dropout)
        
        self.linear2 = torch.nn.Linear(dim, dim)
        self.act2 = torch.nn.GELU()
        self.drop2 = torch.nn.Dropout(dropout)
        
        self.linear3 = torch.nn.Linear(dim, 1)
        self.act3 = torch.nn.Sigmoid()

    def forward(self, x):
        y_pred = self.linear1(x)
        y_pred = self.act1(y_pred)
        y_pred = self.drop1(y_pred)
        
        y_pred = self.linear2(y_pred)
        y_pred = self.act2(y_pred)
        y_pred = self.drop2(y_pred)
        
        y_pred = self.linear3(y_pred)
        y_pred = self.act3(y_pred)
        return y_pred
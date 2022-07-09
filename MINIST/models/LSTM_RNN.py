import torch.nn as nn
import torch

'''
Source Git repository: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02
-intermediate/recurrent_neural_network/main.py
This LSTM RNN model reduces the risk of gradient vanishing the RNN has. 
Editor: Michelle Lu
'''

class RNN(nn.Module):
    def __init__(self, N_INPUTS, N_NEURONS, N_LAYERS, N_OUTPUTS, device):
        # initalize the class
        super(RNN, self).__init__()
        self.device = device
        self.N_NEURONS = N_NEURONS # number of hidden units
        self.N_LAYERS = N_LAYERS # number of layers
        # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size
        self.lstm = nn.LSTM(N_INPUTS, N_NEURONS, N_LAYERS, batch_first=True)
        self.fc = nn.Linear(N_NEURONS, N_OUTPUTS) # fully connected layer

    def forward(self, x):
        # initialize hidden (neural units) and cell states (zeros)
        h0 = torch.zeros(self.N_LAYERS, x.size(0), self.N_NEURONS).to(self.device)
        c0 = torch.zeros(self.N_LAYERS, x.size(0), self.N_NEURONS).to(self.device)

        # Forward LSTM
        # out: tensor(batch_size, N_STEPS, N_NEURONS), h0 and c0 consist of the number of layers,
        # batch and the hidden units (N_Neurons)
        out, _ = self.lstm(x, (h0, c0))

        # return the final state of the last step
        out = self.fc(out[:, -1, :])
        return out

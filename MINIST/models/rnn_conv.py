import torch.nn as nn
import torch

'''
Lab code given and used to compare the effects of a basic RNN model with a LSTM model
LIMITED CHANGES HAVE BEEN MADE TO PRESERVE THE MODEL FOR COMPARISION AGAINST THE LSTM MODEL

RNNs are a class of artificial neural network where connections between units form a directed graph 
along a sequence. This means RNN networks use their own internal memory to be able to process inputs as a sequence as
opposed to each image becoming a separate input

The BasicRNN class initialises the batch_size, n_steps, n_inputs, n_neurons, n_outputs, device, 
builds the basic Rnn layer, followed by a fully connected layer.

The function init_hidden prepares a tensor filled with zeros with the (batch, neuron and ) input sizes and sends 
them to the device (Cuda) 


Editor: 
'''

class BasicRNN(nn.Module):
	def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs, device):
		super(BasicRNN, self).__init__()
		self.device = device
		self.n_neurons = n_neurons   # Hidden
		self.batch_size = batch_size
		self.n_steps = n_steps    # 64
		self.n_inputs = n_inputs  # 28
		self.n_outputs = n_outputs  # 10
		# Basic RNN layer
		self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons)
		# Followed by a fully connected layer
		self.FC = nn.Linear(self.n_neurons, self.n_outputs)

	def init_hidden(self, ):
		# (num_layers, batch_size, n_neurons)
		return (torch.zeros(1, self.batch_size, self.n_neurons)).to(self.device)

	def forward(self, X):
		# transforms X to dimensions: n_steps X batch_size X n_inputs
		# 28 * 64 * 28
		X = X.permute(1, 0, 2)
		self.batch_size = X.size(1)
		self.hidden = self.init_hidden()
		lstm_out, self.hidden = self.basic_rnn(X, self.hidden)
		out = self.FC(self.hidden)
		# Output from fully connected layer  directly
		return out.view(-1, self.n_outputs)  # batch_size X n_output
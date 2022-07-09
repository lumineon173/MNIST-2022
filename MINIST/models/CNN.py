import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
Source Git repository: https://github.com/floydhub/mnist-ConvNet.py
Editor: Matthew Wu
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # defining layers
        # 1: input channals 10: output channels, 5: kernel size, 1: stride
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        # Fully connected layer: input size, output size
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)

    # define forawrd function
    def forward(self, x):
        # pass data into first convolution layer
        x = F.relu(self.conv1(x))
        # pooling kernel size 2, stride 2
        x = F.max_pool2d(x, 2, 2)
        # pass data into second convolution layer
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # flattening images
        x = x.view(-1, 320)
        # first layer of 320 nodes will connect to second layer of 60 nodes
        x = F.relu(self.fc1(x))
        # second layer of 60 nodes will connect to last layer of 10 nodes
        x = self.fc2(x)
        # logarithm of Softmax function
        return F.log_softmax(x, dim=1)
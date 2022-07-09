from __future__ import print_function
import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.CNN import Net
from models.MLP import MLP
from models.rnn_conv import BasicRNN
from models.LSTM_RNN import RNN
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.utils.data as data
from sklearn.metrics import confusion_matrix


# functions to show an image
def imsave(img):
    npimg = img.numpy()
    npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
    im = Image.fromarray(npimg)
    im.save("./results/your_file.jpeg")


def calculate(actual, predicted):

    # initialise true positives, false positives, true negatives, and false negatives
    truepositive = 0
    falsepositive = 0
    truenegative = 0
    falsenegative = 0

    #
    for i in range(len(predicted)):
        # calculate number of true positives
        if actual[i] == predicted[i] == 1:
            truepositive += 1
        # calculate number of false positives
        if predicted[i] == 1 and actual[i] != predicted[i]:
            falsepositive += 1
        # calculate number of false negatives
        if actual[i] == predicted[i] == 0:
            truenegative += 1
        # calculate number of false negatives
        if predicted[i] == 0 and actual[i] != predicted[i]:
            falsenegative += 1

    return truepositive, falsepositive, truenegative, falsenegative


def train_cnn(log_interval, model, device, train_loader, optimizer, epoch, loss_list, acc_list, train_counter):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        loss_list.append(loss.item())
        optimizer.step()

        # set up for learning curve
        total = target.size(0)
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == target).sum().item()
        acc_list.append(correct / total)
        train_counter.append(
            (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def train_mlp(log_interval, model, device, train_loader, optimizer, epoch, loss_list, acc_list, train_counter):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # loss functions  # forward + backward + optimize
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

        # set up for learning curve
        total = target.size(0)
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == target).sum().item()
        acc_list.append(correct / total)
        train_counter.append(
            (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def train_rnn(log_interval, model, device, train_loader, optimizer, epoch, loss_list, acc_list, train_counter):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # reset hidden states
        model.hidden = model.init_hidden()
        data = data.view(-1, 28, 28)
        outputs = model(data)
        # loss  # forward + backward + optimize
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, target)
        loss_list.append(loss.item())
        loss.backward();
        optimizer.step()

        # set up for learning curve
        total = target.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == target).sum().item()
        acc_list.append(correct / total)
        train_counter.append(
            (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def train_LRNN(log_interval, model, device, train_loader, optimizer, epoch, loss_list, acc_list, train_counter):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, 28, 28)
        # forward pass
        outputs = model(data)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, target)
        loss_list.append(loss.item())
        # backward and optimise
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # set up for learning curve
        total = target.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == target).sum().item()
        acc_list.append(correct / total)
        train_counter.append(
            (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()  # disables any dropout or batch normalization in model
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # preparation
            data = torch.squeeze(data)
            data, target = data.to(device), target.to(device)
            # test
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # Initialize the prediction and label lists(tensors)
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = torch.squeeze(data)
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            # Append batch prediction results
            predlist = torch.cat([predlist, preds.view(-1).cpu()])
            lbllist = torch.cat([lbllist, target.view(-1).cpu()])

    # Confusion matrix
    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    print(conf_mat)

    # Evaluation
    TP, FP, TN, FN = calculate(lbllist.numpy(), predlist.numpy())
    precision = TP / (TP + FP)
    print('Prediction = %f' % precision)
    recall = TP / (TP + FN)
    print('Recall = %f' % recall)
    f1 = 2 * ((precision * recall) / (precision + recall))
    print('F1-score: %f' % f1)


def main():
    epoches = 10
    gamma = 0.7
    log_interval = 10
    torch.manual_seed(1)
    save_model = True  # select to save and plot accuracy and loss
    Normalize = False

    # RNNa
    MLP_ = False
    B_RNN = True # basic RNN given in lab (change other RNN basis to false)
    L_RNN = False # LSTM RNN (change other RNN basis to false)

    N_STEPS = 28
    N_INPUTS = 28
    N_NEURONS = 150
    N_OUTPUTS = 10
    N_LAYERS = 2

    # Check whether you can use Cuda
    use_cuda = torch.cuda.is_available()
    # Use Cuda if you can
    device = torch.device("cuda" if use_cuda else "cpu")

    loss_list = []
    acc_list = []
    train_counter = []

    ######################3   Torchvision    ###########################3
    # Use data predefined loader
    # Pre-processing by using the transform.Compose
    # divide into batches
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if Normalize:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('../data', download=True, train=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, **kwargs)
        test_dataset = datasets.MNIST('../data', train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=64, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=1000, shuffle=True, **kwargs)

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    # img = torchvision.utils.make_grid(images)
    # imsave(img)

    # #####################    Build your network and run   ############################
    # Building Model
    if B_RNN:  # Basic RNN
        model = BasicRNN(64, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS, device).to(device)
    elif L_RNN:  # LSTM RNN
        model = RNN(N_INPUTS, N_NEURONS, N_LAYERS, N_OUTPUTS, device).to(device)
    elif MLP_:  # multilayer perceptron
        model = MLP().to(device)
    else:  # CNN
        model = Net().to(device)

    # Building the Optimiser
    if B_RNN:
        optimizer = optim.Adadelta(model.parameters(), lr=0.01)
    elif L_RNN:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif MLP_:
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # changed optimiser from SGD to adam for better results
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epoches + 1):
        if B_RNN:
            train_rnn(log_interval, model, device, train_loader, optimizer, epoch, loss_list, acc_list, train_counter)
        elif L_RNN:
            train_LRNN(log_interval, model, device, train_loader, optimizer, epoch, loss_list, acc_list, train_counter)
        elif MLP_:
            train_mlp(log_interval, model, device, train_loader, optimizer, epoch, loss_list, acc_list, train_counter)
        else:
            train_cnn(log_interval, model, device, train_loader, optimizer, epoch, loss_list, acc_list, train_counter)

        test(model, device, test_loader)
        scheduler.step()

    # Save model selection allows the model to be saved and plots the accuracy and loss of the testing algorithm
    # LEARNING CURVE
    if save_model:
        torch.save(model.state_dict(), "./results/mnist_cnn.pt")

        #learning Curve
        fig, ax = plt.subplots()
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel('Loss', color='blue')
        ax.plot(train_counter, np.array(loss_list), color='blue')
        # plots the accuracy on the same graph
        ay = ax.twinx()
        ay.set_ylabel('Accuracy', color='red')
        ay.set_ylim([0, 1])  # sets y axis range to 1
        ay.plot(train_counter, np.array(acc_list), color='red')
        # fig.tight_layout()
        plt.show()
        # close the image to end have the process finish with exit code or it will still keep running as the image
        # is open


if __name__ == '__main__':
    main()

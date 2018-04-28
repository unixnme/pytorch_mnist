import math
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

class Net(nn.Module):
    def __init__(self, ndim=2, last_layer=True):
        super(Net, self).__init__()
        self.last_layer = last_layer
        self.ndim = ndim
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) # 26
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3) # 24 --> 12
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3) # 10
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3) # 8 --> 4
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3) # 2
        self.conv5_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 128, bias=False)
        self.fc2 = nn.Linear(128, 32, bias=False)
        self.fc3= nn.Linear(32, ndim, bias=False)
        self.softmax = AngleSoftmax(10)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(F.max_pool2d(self.conv2(x), 2))
        x = F.elu(self.conv3(x))
        x = F.elu(F.max_pool2d(self.conv4(x), 2))
        x = F.elu(self.conv5_drop(self.conv5(x)))
        x = x.view(-1, 512)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        if self.last_layer is True:
            x = self.softmax(x)
        return x

class AngleSoftmax(nn.Module):
    def __init__(self, out_feature):
        super(AngleSoftmax, self).__init__()
        self.out_feature = out_feature
        self.weight = Parameter(torch.Tensor(1, out_feature))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        u = torch.cos(self.weight)
        v = torch.sin(self.weight)
        w = torch.cat([u, v], dim=0)
        x = x.view(-1,2,1) - w.view(1,2,-1)
        x = -torch.tanh(torch.sum(x**2, dim=1))
        return F.log_softmax(x, dim=1)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            pass

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def plot_features(model):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.cpu().data.numpy()
            target = target.cpu().data.numpy()
            for label in range(10):
                idx = target == label
                plt.scatter(output[idx,0], output[idx,1])
            plt.legend(np.arange(10, dtype=np.int32))
            plt.show()
            break

if __name__ == '__main__':
    model = Net(2).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    model_file = 'model.h5'
    if not os.path.isfile(model_file):
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test()
        torch.save(model, model_file)
    else:
        model = torch.load(model_file)

    feature_model = Net(2, last_layer=False)
    feature_model.load_state_dict(model.state_dict())
    feature_model.to(device)
    plot_features(feature_model)

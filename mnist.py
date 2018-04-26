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
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
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
    def __init__(self, ndim=2):
        super(Net, self).__init__()
        self.ndim = ndim
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(10, affine=True)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(20, affine=True)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=3)
        self.conv4_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, ndim)
        self.softmax = AngleSoftmax(10)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.bn1(x)
        x = F.elu(F.max_pool2d(self.conv2(x), 2))
        x = F.elu(self.conv3(x))
        x = self.bn2(x)
        x = F.elu(F.max_pool2d(self.conv4_drop(self.conv4(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.softmax(x)
        return x

class FeatureExtractor(Net):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__(model.ndim)
        self.load_state_dict(model.state_dict())

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.bn1(x)
        x = F.elu(F.max_pool2d(self.conv2(x), 2))
        x = F.elu(self.conv3(x))
        x = self.bn2(x)
        x = F.elu(F.max_pool2d(self.conv4_drop(self.conv4(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
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
        x = x.mm(w)
        return F.log_softmax(x)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def plot_features(model):
    model.eval()
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
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
    model = Net(2)
    if args.cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    model_file = 'model.h5'
    if not os.path.isfile(model_file):
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test()
        torch.save(model, model_file)
    else:
        model = torch.load(model_file)

    feature_model = FeatureExtractor(model)
    if args.cuda:
        feature_model.cuda()
    plot_features(feature_model)

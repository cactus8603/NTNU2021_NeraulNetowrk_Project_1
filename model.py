import pandas as pd
import numpy as np 
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

from preprocess import *
from data import *

import matplotlib.pyplot as plt

class GRU(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, n_layers):
        super(GRU, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.dropout = dropout

        self.gru = torch.nn.GRU(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_dim, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, output_dim)
        self.relu = torch.nn.ReLU(inplace=False)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):

        output, hidden = self.gru(x)
        output = self.relu(output)
        output = self.fc1(output)
        # output = self.relu(output)
        output = self.fc2(output)
        # output = self.relu(output)
        output = self.fc3(output)
        # output = self.softmax(output)
        output = output[:, -1, :]

        return output

def spilt_data(BATCH_SIZE):
    x, y = getdata(BATCH_SIZE)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    # train_x, test_x, train_y, test_y = torch.from_numpy(train_x), torch.from_numpy(test_x), torch.from_numpy(train_y), torch.from_numpy(test_y)
    # return train_x, test_x, train_y, test_y
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_x, train_y = torch.from_numpy(train_x).to(device), torch.from_numpy(train_y).to(device)
    test_x, test_y = torch.from_numpy(test_x).to(device), torch.from_numpy(test_y).to(device)
    # return train_x, train_y
    # print(train_x.device)
    torch_train_dataset = data.TensorDataset(train_x, train_y)
    torch_test_dataset = data.TensorDataset(test_x, test_y)

    trainloader = data.DataLoader(
        dataset=torch_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        # num_workers=2
        worker_init_fn=seed_worker
    )
    """
    cuda0 = torch.device('cuda:0')  
    for i, x in enumerate(trainloader):
        x = x.to(cuda0)
        # y = y.to(cude0)
    """
    testloader = data.DataLoader(
        dataset=torch_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        worker_init_fn=seed_worker
    )
    
    return trainloader, testloader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':

    torch.manual_seed(0)
    BATCH_SIZE = 32
    lr = 3e-4
    EPOCH = 300

    trainloader, testloader = spilt_data(BATCH_SIZE)
    # train_x, test_x, train_y, test_y = spilt_data(lookforward=5)
    # train_x, train_y = spilt_data(BATCH_SIZE)
    
    # print(train_x)
    # print(train_x.size())
    
    model = GRU(
        input_dim=7,
        hidden_dim=32, 
        output_dim=7, 
        dropout=0, 
        n_layers=3
        )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.double()
    # print(model)

    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    train_loss = []
    train_acc = []
    
    for epoch in range(EPOCH):
        # print('\n train Epoch: %d' % epoch)
        # model.train()
        train_loss_tmp = 0
        train_loss_avg = 0
        correct = 0
        total = 0
        loss = 0
        # inputs, targets = train_x.to(device), train_y.to(device)
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # print(inputs)
            # print(targets.shape)
            # print(targets)
            # if (batch_idx==10):break
            # print(batch_idx)
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(outputs)
            outputs.to(device)
            # print(outputs.device)
            # print(targets.device)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # print(loss.item())

            # train_loss_tmp += loss.item()
            # print(targets)
            # print(targets.shape)
            # print(outputs)
            # print(output.shape)
            # print(torch.max(outputs, 1))
            _, predicted = torch.max(outputs, 1)
            # print(predicted)
            # print(targets)
            # total += targets.size(0)
            # print(torch.eq(targets, predicted))
            # correct += predicted.eq(targets).sum().item()
            # train_loss_avg = train_loss_tmp / (batch_idx + 1)
        print("mode:training loss:", loss.item() )
        train_loss.append(loss.item())
        # train_acc.append(100. * correct / total)
        # print('\n -----train Epoch Over: %d------\n' % epoch)
        # print(len(trainloader), 'Loss: %.8f | Acc: %.8f%% (%d/%d)'
            # % (train_loss_avg, 100. * correct / total, correct, total))
        # print(output)
        """
        inputs, targets = next(iter(trainloader))
        
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
        print(loss)
       """
    # np.delete(train_loss, (0,1))
    # finish training
    model.eval()
    test_loss = []
    for epoch in range(EPOCH): 
        with torch.no_grad():
            loss = 0
            outputs = 0
            for batch_idx, (inputs, targets) in enumerate(testloader):
                outputs = model(inputs)
                outputs.to(device)
                loss = criterion(outputs, targets)
            print("mode:eval loss:", loss.item() )
            test_loss.append(loss.item())
    
    train_loss = np.delete(train_loss, (0,1,2,3))
    test_losee = np.delete(test_loss, (0,1,2,3))
    plt.plot(train_loss, 'r', label='train_loss')
    # plt.plot(test_loss, 'b', label='test_loss')
    plt.show()
    plt.savefig("out2.jpg")

    



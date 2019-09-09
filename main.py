import numpy as np
import dataloader
import torch
from torch import optim
import torch.nn as nn
from model import CNNLSTM
from dataset import imageDataset
class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model, num_updates):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                decay = min(self.mu, (1+num_updates)/(10+num_updates))
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]
def train(dataset, model, optimizer, start):
    #print(model)
    model.train()
    losses = []
    criterion = nn.MSELoss()
    for i, (label, images) in enumerate(dataset):
        #for b in range(images.size(0)):
        #    print(images[b,:,0,0,0])
        x = images.cuda()
        #print(x)
        y = label.cuda()
        optimizer.zero_grad()
        ans = model(x).squeeze(1)
        #print(y)
        #print(ans)
        #loss = torch.mean((ans-y)**2)
        loss = criterion(ans, y)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()
        #ema(model, i+start*len(dataset))
        losses.append(loss.item())
        #print(label)
        #print(images.size())
        print("\rSTEP {:8d}/{} loss {:8f}".format(i + 1, len(dataset), loss.item()), end='')
    loss_avg = np.mean(losses)
    print("STEP {:8d} Avg_loss {:8f}\n".format(start, loss_avg))
def valid(dataset, model, testdataset, iter):
    model.eval()
    criterion = nn.MSELoss()
    losses = []
    for i, (label, images) in enumerate(dataset):
        x = images.cuda()
        y = label.cuda()
        ans = model(x).squeeze(1)
        loss = criterion(ans, y)
        losses.append(loss.item())
    loss_avg = np.mean(losses)
    print("VALID Avg_loss {:8f}\n".format(loss_avg))

    fp = open("64"+"test"+str(iter)+".txt", "w")
    for i, (label, images) in enumerate(testdataset):
        x = images.cuda()
        ans = model(x).squeeze(1)
        for n in ans:
            fp.write(str(n.item())+ '\n')
    fp.close()



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = imageDataset('train')
    vdataset = imageDataset('valid')
    tdataset = imageDataset('test')
    valid_dataset = dataloader.get_loader(vdataset, 4, shuffle = False)
    train_dataset = dataloader.get_loader(dataset, 4, shuffle = True)
    test_dataset = dataloader.get_loader(tdataset, 4, shuffle = False)
    model = CNNLSTM().to(device)
    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.SGD(lr=0.00001, params=model.parameters(), momentum = 0.9)
    #for name, param in model.named_parameters():
    #    print(name, param.requires_grad)
    #ema = EMA(0.999)
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        if 'conv' in name:
    #            param.requires_grad = False
    #for name, param in model.named_parameters():
    #    print(name, param.requires_grad)
    valid(valid_dataset, model, test_dataset, 30)
    for iter in range(20):
        train(train_dataset, model, optimizer, iter)
        #ema.assign(model)
        valid(valid_dataset, model, test_dataset, iter)
        #ema.resume(model)

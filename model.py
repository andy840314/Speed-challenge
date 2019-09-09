import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision import models
from torch.autograd import Variable
class CNNLSTM(nn.Module):
    def __init__(self, num_layers = 2, hidden_size = 256):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        vgg = nn.Sequential(*list(models.vgg16_bn(pretrained = True).children())[0])
        self.conv = nn.Sequential(*list(vgg.children())[0:14])
        #self.mpool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0, dilation=1, ceil_mode=False)
        #self.con2d = nn.Conv2d(128, 16, kernel_size = (3,3), stride = (2, 2), padding=0, bias=True)
        #self.alet = nn.Sequential(*list(models.alexnet(pretrained = True).children()))
        alexnet = nn.Sequential(*list(models.alexnet(pretrained = True).children())[0])
        self.conv2 = nn.Sequential(*list(alexnet.children()))
        self.p = nn.AvgPool2d((16, 50), stride=(1, 1))
        #self.rnnfc1 = nn.Linear(1536, 1000)
        #self.rnnfc2 = nn.Linear(5000, 500)
        self.lstm = nn.LSTM(input_size = 128, hidden_size = hidden_size, dropout = 0.2, num_layers = num_layers, bidirectional = False, batch_first = False)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x):
        batch_size, timesteps = x.size()[0], x.size()[2]
        #h0, c0 = self.init_hidden(batch_size)
        img_feats = []
        for t in range(timesteps):
            img_feat = self.conv(x[:,:,t,:,:])
            img_feat = self.p(img_feat)
            #print(img_feat.size())
            #img_feat = F.relu(self.con2d(img_feat))
            #img_feat = F.dropout(img_feat, p=0.2, training=self.training)
            #img_feat = self.mpool(img_feat)
            #print(img_feat.size())
            img_feat = img_feat.view(batch_size, -1)
            #print(img_feat)
            #print(img_feat.size())
            img_feats.append(img_feat)
        img_feats = torch.stack(img_feats, dim=0)
        #print(img_feats.size())
        #img_feats.permute(1, 0, 2)
        #img_feats = F.relu(self.con2d(img_feats))
        #img_feats = F.dropout(img_feats, p=0.2, training=self.training)
        #img_feats = self.rnnfc2(img_feats)
        #state = self._init_state(batch_size)
        lstm, _ = self.lstm(img_feats)
        #print(lstm)
        #out = F.leaky_relu(self.fc1(lstm[-1]), negative_slope=0.1)
        out = F.relu(self.fc1(lstm[-1]))
        out = F.dropout(out, p=0.2, training=self.training)
        out = self.fc2(out)
        #print(out)
        return out
    def _init_state(self, batch_size):
        weight = next(self.parameters()).data
        return (
                Variable(weight.new(self.num_layers, batch_size, self.hidden_size).normal_(0.0, 0.01)),
                Variable(weight.new(self.num_layers, batch_size, self.hidden_size).normal_(0.0, 0.01))
                )
        #hidden = Variable(next(self.parameters()).data.new(batch_size, self.num_layers, self.hidden_size))
        #cell = Variable(next(self.parameters()).data.new(batch_size, self.num_layers, self.hidden_size))
        #return hidden, cell

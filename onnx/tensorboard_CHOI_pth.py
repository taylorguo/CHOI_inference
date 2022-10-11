# tensorboard --logpath=D:\log --port=6006

import torch
import torchvision

from torch.autograd import Variable
from tensorboardX import SummaryWriter

input = Variable(torch.rand(1, 3, 1200, 700))

model = torchvision.Model.resnet18()

writer = SummaryWriter(logdir="./log", comment="resnet18")
with writer:
    writer.add_graph(model, (input,))
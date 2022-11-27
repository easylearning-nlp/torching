import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_sequence = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model_sequence(x)
        return x

net = Model()
print(net)

input = torch.ones((64, 3, 32, 32))
output = net(input)
print(output.shape)

writer = SummaryWriter("logs_seq")
writer.add_graph(net, input)
writer.close()
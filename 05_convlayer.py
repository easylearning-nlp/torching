import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset=dataset, batch_size=64)

class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

net = CnnModel()
step = 0
writer = SummaryWriter("cnnlogs")
for data in dataloader:
    imgs, targets = data
    output = net(imgs)
    #print(imgs.shape)
    #print(output.shape)
    writer.add_images("input", imgs, step)

    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    writer.close()

    step = step+1

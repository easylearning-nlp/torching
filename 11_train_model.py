import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model import *


train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

train_dataloader = DataLoader(dataset=train_data, batch_size=64)
test_dataloader = DataLoader(dataset=test_data, batch_size=64)


#创建模型
net = Model()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

#训练过程
total_train_step = 0
total_test_step = 0
epoch = 10

writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("--------------第{}轮开始--------------".format(i+1))
    for data in train_dataloader:
        imgs, targets = data
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    total_test_loss = 0
    with torch.no_grad():
        for testdata in test_dataloader:
            test_imags, test_target = testdata
            test_outputs = net(test_imags)
            loss = loss_fn(test_outputs, test_target)
            total_test_loss = total_test_loss + loss.item()

    print("整体测试集上的Loss:{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(net,"net_{}.pth".format(i+1))
    print("模型已经保存")

writer.close()


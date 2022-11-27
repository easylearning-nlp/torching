import torchvision
import torch

vgg16 = torchvision.models.vgg16(pretrained=False)

torch.save(vgg16, "vgg16_method_1.pth")

# 方式2 保存是的模型的参数
torch.save(vgg16.state_dict(), "vgg16_method_2.pth")
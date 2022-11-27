import torch
import torchvision

# 方式1来加载模型


model_1 = torch.load("/Users/easylearninghow/Desktop/Torching/vgg16_method_1.pth")
#print(model_1)


vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("/Users/easylearninghow/Desktop/Torching/vgg16_method_2.pth"))
print(vgg16)
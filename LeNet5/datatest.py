import torch
from torchvision import transforms, datasets

import torchvision
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((1/12.75), (10/12.75)),
    # transforms.Normalize((0.13066047430038452,), (0.30810782313346863,)),
])

batch_size=60000
#Data set
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transform,
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


for batch_idx, data in enumerate(train_loader, 0):
    inputs, targets = data #inpus为所有训练样本
    x=inputs.view(-1,16*16) #将（6000，1，28，28）大小的inputs转换为（60000，28*28）的张量
    x_std=x.std().item() #计算所有训练样本的标准差
    x_mean=x.mean().item() #计算所有训练样本的均值
    x_min = x.min()
    x_max = x.max()
    print(x_min,x_max,x_mean,x_std)

# for num, (image, label) in enumerate(train_loader):
#     image_batch = torchvision.utils.make_grid(image, padding=2)
#     plt.imshow(np.transpose(image_batch.numpy(), (1, 2, 0)), vmin=0, vmax=255)
#     plt.show()
#     print(label)
#     if num==2:
#         break


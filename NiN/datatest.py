import torch
from torchvision import transforms, datasets

import torchvision
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4913996756076813,0.48215845227241516,0.44653093814849854),
        (0.06055857613682747,0.06115545332431793,0.06767884641885757)),
])

batch_size=50000
#Data set
train_dataset = datasets.CIFAR10(root='./data',
                               train=True,
                               transform=transform,
                               download=True)

test_dataset = datasets.CIFAR10(root='./data',
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
    print(inputs.size())
    x=inputs.view(50000, 3, 32*32)
    print(x.size())
    x_std=x.std(-1).std(0) #计算所有训练样本的标准差
    x_mean=x.mean(-1).mean(0) #计算所有训练样本的均值
    x_min = x.min()
    x_max = x.max()
    print(x_std[0].item())
    print(x_std[1].item())
    print(x_std[2].item())

# for num, (image, label) in enumerate(train_loader):
#     image_batch = torchvision.utils.make_grid(image, padding=2)
#     plt.imshow(np.transpose(image_batch.numpy(), (1, 2, 0)), vmin=0, vmax=255)
#     plt.show()
#     print(label)
#     if num==2:
#         break


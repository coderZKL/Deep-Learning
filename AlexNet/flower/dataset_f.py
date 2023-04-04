import torch
import os,json
from torchvision import transforms, datasets

import torchvision
import matplotlib.pyplot as plt
import numpy as np


data_transform = {
        "train": transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    # transforms.RandomResizedCrop(224),
                                    #  transforms.RandomHorizontalFlip(),
                                    #  transforms.ToTensor(),
         #                             transforms.Normalize(
         # (0.4668380916118622, 0.4245220124721527, 0.30413004755973816),
         #  (0.06382322311401367, 0.05896018445491791, 0.08515258133411407))
                                     ]),
        "val": transforms.Compose([
                                    transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ])}


image_path = os.path.join(os.getcwd(), "flower_data")
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                        transform=data_transform["train"])
validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])
train_num = len(train_dataset)
val_num = len(validate_dataset)


batch_size = 3306
train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            )

validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                )

print("using {} images for training, {} images for validation.".format(train_num,
                                                                        val_num))


# for batch_idx, data in enumerate(train_loader, 0):
#     inputs, targets = data #inpus为所有训练样本
#     print(inputs.size())
#     x=inputs.view(3306, 3, 224*224) #将（6000，1，28，28）大小的inputs转换为（60000，28*28）的张量
#     print(x.size())
#     x_mean=x.mean(-1).mean(0) #计算所有训练样本的均值
#     x_std=x.std(-1).std(0)
#     print(x_std[0].item())
#     print(x_std[1].item())
#     print(x_std[2].item())
#     print(x.max())
#     print(x.min())

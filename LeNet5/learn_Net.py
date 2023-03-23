import torch
from torchvision import transforms, datasets
from torch import nn, optim

from leNet_5 import Subsampling, MapConv, RBFLayer, LeNet5

def data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((1/12.75), (10/12.75)),
        # transforms.Normalize((0.13066047430038452,), (0.30810782313346863,)),
    ])

    batch_size=16
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
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    return train_loader, test_loader

def net():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = data()
    C1 = nn.Conv2d(1, 6, 5, padding = 2, padding_mode='replicate').to(device)
    S2 = Subsampling(6).to(device)
    C3 = MapConv(6, 16, 5).to(device)
    S4 = Subsampling(16).to(device)
    C5 = nn.Conv2d(16, 120, 5).to(device)
    F6 = nn.Linear(120, 84).to(device)
    Output = RBFLayer(84, 10).to(device)

    for i, (image, label) in enumerate(train_loader):
        x = image.to(device)
        out = C1(x)
        out = S2(out)
        out = C3(out)
        out = S4(out)
        out = C5(out)
        out = out.view(-1, 120)
        out = F6(out)
        out = Output(out)
        print(out)
        break

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = data()
    lenet = LeNet5(device)
    for i, (image, label) in enumerate(train_loader):
        x = image.to(device)
        out = lenet(x)
        print(out)
        break

def main():
    # net()
    train()

if __name__ == "__main__":
    main()

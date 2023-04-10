import json, os
import torch
from torchvision import transforms, datasets
from torch import nn, optim
from torch.utils.data import DataLoader

from tools import Logger


def nin_block(in_channels,out_channels,kernel_size,stride,padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size, 
                    stride= stride,padding = padding),nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU(),
    )

def nin():
    net = nn.Sequential(
    nin_block(3,96,kernel_size= 5,stride=1,padding= 2),
    nn.MaxPool2d(kernel_size= 3,stride=2),
    nin_block(96,256,kernel_size= 3,stride=1,padding= 2),
    nn.MaxPool2d(kernel_size= 3,stride=2),
    nin_block(256,384,kernel_size= 3,stride=1,padding= 1),
    nn.MaxPool2d(kernel_size= 3,stride=2),
    nn.Dropout(0.5),
    nin_block(384,10,kernel_size= 3,stride=1,padding= 1),
    nn.AdaptiveAvgPool2d(output_size=(1,1)),
    nn.Flatten()
    )
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    return net

class NiNMoudle():
    def __init__(self, batch_size, epoch_num):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.net = nin().to(self.device)
        self.epoch_num = epoch_num
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), 
                        lr = 1.0e-2, momentum=0.9, weight_decay=5.0e-4)
        # self.optimizer = optim.SGD(self.net.parameters(), lr = 0.1)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                        mode='min', factor=0.1, patience=1)

        self.lossList = []
        self.testAcc = []
        # self.trainAcc = []
        self.trainnum = 0
        self.testnum = 0

    def data(self):
        transform = transforms.Compose([
            # transforms.Resize(224),
            transforms.ToTensor(),
            # transforms.Normalize(
            # (0.4913996756076813,0.48215845227241516,0.44653093814849854),
            # (1.0,1.0,1.0)),
            transforms.Normalize(
            (0.4913996756076813,0.48215845227241516,0.44653093814849854),
            (0.06055857613682747,0.06115545332431793,0.06767884641885757)),
        ])
        #Data set
        train_dataset = datasets.CIFAR10(root='./data',
                                    train=True,
                                    transform=transform,
                                    download=True)
        test_dataset = datasets.CIFAR10(root='./data',
                                    train=False,
                                    transform=transform)
        # Data loader
        self.train_loader = DataLoader(dataset=train_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=False)
        self.trainnum = len(train_dataset)
        self.trainbatch = len(self.train_loader)
        self.testnum = len(test_dataset)
        self.testbatch = len(self.test_loader)

    def train(self):
        for epoch in range(self.epoch_num):
            lossSum = 0.0
            log.info('epoch: {:02d} / {:d}'.format(epoch+1, self.epoch_num))

            self.net.train()
            for i, (image, label) in enumerate(self.train_loader):
                x = image.to(self.device)
                y = label.to(self.device)

                #梯度下降与参数更新
                self.optimizer.zero_grad()
                out = self.net(x)
                loss = self.loss(out, y)
                loss.backward()
                self.optimizer.step()

                lossSum += loss.item()
                if (i+1)%20==0:
                    log.info('batch:{:d}/{:d} lossAver:{:f}'
                                .format(i+1, self.trainbatch, lossSum/(i+1)))
                # print(loss.item())
                # progress_bar(epoch*self.trainnum+i, self.epoch_num*self.trainnum)
                # break

            self.net.eval()
            self.lossList.append(lossSum / self.trainbatch)
            if self.scheduler is not None:
                self.scheduler.step(lossSum / self.trainbatch)
            log.info('lossList:'+str(self.lossList))
            self.test()
        torch.save(self.net.state_dict(), './module.pth')

    def test(self):
        #测试部分，每训练一个epoch就在测试集上进行错误率的求解与保存
        with torch.no_grad():
            accNum = 0
            for img, label in self.test_loader:
                x = img.to(self.device)
                label = label.to(self.device)
                out = self.net(x)
                pred_y = out.argmax(dim = 1)
                accNum += int((pred_y==label).sum())
            self.testAcc.append(accNum / self.testnum)
        
        log.info('testAcc:'+str(self.testAcc))

        
def main():
    m = NiNMoudle(128, 100)
    m.data()
    m.train()

    with open(os.path.join(os.getcwd(), 'result.json'),'w',encoding='utf8') as f:
        json.dump([m.lossList, m.testAcc], f, ensure_ascii=False)

if __name__ == "__main__":
    log = Logger('run.log')
    main()


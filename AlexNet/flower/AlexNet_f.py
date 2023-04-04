import json, os
import torch
from torchvision import transforms, datasets
from torch import nn, optim
from torch.utils.data import DataLoader

from tools import Logger, progress_bar


# 局部响应归一化层
class LRN(nn.Module):
    def __init__(self, in_channels: int, device, k=2, n=5, alpha=1.0e-4, beta=0.75):
        super(LRN, self).__init__()
        self.in_channels = in_channels
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.device = device

    def forward(self, x):
        tmp = x.pow(2)
        div = torch.zeros(tmp.size()).to(self.device)
        
        for batch in range(tmp.size(0)):
            for channel in range(tmp.size(1)):
                st = max(0, channel - self.n // 2)
                ed = min(channel + self.n // 2, tmp.size(1)-1)+1
                div[batch, channel] = tmp[batch, st:ed].sum(dim=0)
        out = x / (self.k + self.alpha * div).pow(self.beta)
        return out


# from tools import progress_bar
class AlexNet(nn.Module):
    # 定义网络结构
    def __init__(self, device, nclass):
        super(AlexNet, self).__init__()
        
        self.C1 = nn.Conv2d(3, 96, 11, stride=4, padding=2)
        # self.C1.bias.data = torch.zeros(self.C1.bias.data.size())
        self.N1 = LRN(96, device=device)

        self.C2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        # self.C2.bias.data = torch.ones(self.C2.bias.data.size())
        self.N2 = LRN(256, device=device)
        
        self.C3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        # self.C3.bias.data = torch.zeros(self.C3.bias.data.size())
        self.C4 = nn.Conv2d(384, 384, 3, stride=1, padding=1)
        # self.C4.bias.data = torch.ones(self.C4.bias.data.size())
        self.C5 = nn.Conv2d(384, 256, 3, stride=1, padding=1)
        # self.C5.bias.data = torch.ones(self.C5.bias.data.size())

        self.F6 = nn.Linear(256*6*6, 4096)
        self.F7 = nn.Linear(4096, 4096)
        self.F8 = nn.Linear(4096, nclass)

        self.pool = nn.MaxPool2d(3, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.act = nn.ReLU(True)

        # for m in self.modules(): #权重以及线性层偏置初始化
            # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            #     m.weight.data = torch.normal(torch.zeros(m.weight.data.size()), 
            #                                 torch.ones(m.weight.data.size()) * 0.01)
            # if isinstance(m, nn.Linear):
            #     m.bias.data = torch.ones(m.bias.data.size())
        
        self.to(device)

    def forward(self, x):
        x = self.pool(self.N1(self.act(self.C1(x))))
        x = self.pool(self.N2(self.act(self.C2(x))))
        x = self.act(self.C3(x))    
        x = self.act(self.C4(x))
        x = self.pool(self.act(self.C5(x)))
        x = x.view(-1, 256*6*6)
        x = self.dropout(self.act(self.F6(x)))
        x = self.dropout(self.act(self.F7(x)))
        x = self.act(self.F8(x))
        return x

class AlexNetMoudle():
    def __init__(self, batch_size, epoch_num):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.net = AlexNet(self.device, 5)
        self.epoch_num = epoch_num
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), 
                        lr = 1.0e-2, momentum=0.9, weight_decay=5.0e-4)
        # self.optimizer = optim.Adam(self.net.parameters(), lr=0.0002)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                        mode='min', factor=0.1, patience=1)

        self.lossList = []
        self.testAcc = []
        self.trainAcc = []
        self.trainnum = 0
        self.testnum = 0

    def data(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4668380916118622, 0.4245220124721527, 0.30413004755973816),
                (1.0, 1.0, 1.0)),
        ])
        
        image_path = os.path.join(os.getcwd(), "flower_data")
        assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
        #Data set
        train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                    transform=transform,)
        test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
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
                if (i+1)%2==0:
                    log.info('batch:{:d}/{:d} lossAver:{:f}'
                                .format(i+1, self.trainbatch, lossSum/(i+1)))

            if self.scheduler is not None:
                self.scheduler.step(lossSum / self.trainbatch)

            self.lossList.append(lossSum / self.trainbatch)
            log.info('lossList:'+str(self.lossList))
            self.test()
        torch.save(self.net.state_dict(), './module.pth')

    def test(self):
        #测试部分，每训练一个epoch就在测试集上进行错误率的求解与保存
        self.net.eval()
        with torch.no_grad():
            accNum = 0
            for img, label in self.train_loader:
                x = img.to(self.device)
                label = label.to(self.device)
                out = self.net(x)
                pred_y = out.argmax(dim = 1)
                # print(pred_y)
                accNum += (pred_y==label).sum().item()
            self.trainAcc.append(accNum / self.trainnum)
        log.info('trainAcc:'+str(self.trainAcc))

        with torch.no_grad():
            accNum = 0
            for img, label in self.test_loader:
                x = img.to(self.device)
                label = label.to(self.device)
                out = self.net(x)
                pred_y = out.argmax(dim = 1)
                accNum += (pred_y==label).sum().item()
            self.testAcc.append(accNum / self.testnum)
        log.info('testAcc:'+str(self.testAcc))

        
def main():
    m = AlexNetMoudle(128, 10)
    m.data()
    m.train()

    with open(os.path.join(os.getcwd(), 'result.json'),'w',encoding='utf8') as f:
        json.dump([m.lossList, m.testAcc], f, ensure_ascii=False)

if __name__ == "__main__":
    log = Logger('run.log')
    main()

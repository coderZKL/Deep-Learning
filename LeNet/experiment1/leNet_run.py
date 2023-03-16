import torch
from torchvision import transforms, datasets
from torch import nn, optim
import json,os

from tools import Logger

# from tools import progress_bar
class LeNet(nn.Module):
    def __init__(self, device):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(12, 12, 5, stride=2, padding=2)
        self.fc1 = nn.Linear(12*4*4, 30)
        self.fc2 = nn.Linear(30, 10)
        self.act = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                F_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data = torch.rand(m.weight.data.size()) * 4.8 / F_in - 2.4 / F_in
            if isinstance(m, nn.Linear):
                F_in = m.in_features
                m.weight.data = torch.rand(m.weight.data.size()) * 4.8 / F_in - 2.4 / F_in

        self.to(device)

    def forward(self, x):
        x = 1.7159*self.act(self.conv1(x)*2.0/3.0)
        x = 1.7159*self.act(self.conv2(x)*2.0/3.0)
        x = x.view(-1, 192)
        x = 1.7159*self.act(self.fc1(x)*2.0/3.0)
        x = 1.7159*self.act(self.fc2(x)*2.0/3.0)
        return x


class LeNetMoudle():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 1
        self.net = LeNet(self.device)
        self.epoch_num = 100
        self.loss = nn.MSELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr = 1.0e-3, momentum=0.9)
        self.lossList = []
        self.testError = []
        self.trainnum = 0
        self.testnum = 0

    def data(self):
        transform = transforms.Compose([
            transforms.Resize((16,16)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            # transforms.Lambda(lambda x:(x*2-1)),
        ])
        #Data set
        train_dataset = datasets.MNIST(root='./data',
                                    train=True,
                                    transform=transform,
                                    download=True)
        test_dataset = datasets.MNIST(root='./data',
                                    train=False,
                                    transform=transform)
        # Data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=False)
        self.trainnum = len(train_dataset)
        self.testnum = len(test_dataset)

    def train(self):
        for epoch in range(self.epoch_num):
            lossSum = 0.0
            for i, (image, label) in enumerate(self.train_loader):
                x = image.to(self.device)

                #将标签转化为one-hot向量s
                y = torch.zeros(1, 10)
                y[0][label] = 1.0
                y = y.to(self.device)

                #梯度下降与参数更新
                out = self.net(x)
                self.optimizer.zero_grad()
                loss = self.loss(out, y)
                loss.backward()
                self.optimizer.step()
                lossSum += loss.item()
                if (i+1)%6000==0:
                    log.info('lossSum:'+str(lossSum))
                # print(loss.item())
                # progress_bar(epoch*self.trainnum+i, self.epoch_num*self.trainnum)

            self.lossList.append(lossSum / self.trainnum)
            log.info('lossList:'+str(self.lossList))
            #测试部分，每训练一个epoch就在测试集上进行错误率的求解与保存
            with torch.no_grad():
                errorNum = 0
                for img, label in self.test_loader:
                    x = img.to(self.device)
                    out = self.net(x)
                    _, pred_y = out.max(dim = 1)
                    label = label.to(self.device)
                    if(pred_y != label): errorNum += 1
                self.testError.append(errorNum / self.testnum)
            log.info('testError:'+str(self.testError))
        torch.save(self.net.state_dict(), './module.pth')

def main():
    m = LeNetMoudle()
    m.data()
    m.train()

    with open(os.path.join(os.getcwd(), 'result.json'),'w',encoding='utf8') as f:
        json.dump([m.lossList, m.testError], f, ensure_ascii=False)

if __name__ == "__main__":
    log = Logger('run.log')
    main()

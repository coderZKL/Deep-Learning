import torch
from torchvision import transforms, datasets
from torch import nn, optim
import json,os

import sys 
sys.path.append("..")
from tools import Logger, RBF_WEIGHT, progress_bar


# 损失函数
def loss_fn(pred, label):
    if(label.dim() == 1):
        losses = pred[torch.arange(pred.size(0)), label]
    else:
        losses = pred[torch.arange(pred.size(0)), label.squeeze()]
    return (losses.sum()/(losses.size()[0]))


# 可训练的池化层（采样层）
class Subsampling(nn.Module):
    def __init__(self, in_channel):
        super(Subsampling, self).__init__()
        
        self.pool = nn.AvgPool2d(2)
        self.in_channel = in_channel
        F_in = 4 * self.in_channel
        # 初始化参数并将其列入梯度优化中
        self.weight = nn.Parameter(torch.rand(self.in_channel) * 4.8 / F_in - 2.4 / F_in, requires_grad=True)
        self.bias = nn.Parameter(torch.rand(self.in_channel), requires_grad=True)
    
    def forward(self, x):
        x = self.pool(x)
        #对每一个channel的特征图进行池化，结果存储在这里
        outs = [] 

        for channel in range(self.in_channel):
            #这一步计算每一个channel的池化结果[batch_size, height, weight]
            out = x[:, channel] * self.weight[channel] + self.bias[channel]
            #把channel的维度加进去[batch_size, channel, height, weight]
            outs.append(out.unsqueeze(1)) 
        # 按照channel所在维度进行拼接
        return torch.cat(outs, dim = 1)


# 第三层卷积层
class MapConv(nn.Module):
    def __init__(self, in_channel=6, out_channel=16, kernel_size = 5):
        super(MapConv, self).__init__()
        
        #定义特征图的映射方式
        mapInfo = [[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1],
                   [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                   [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
                   [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
                   [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                   [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]]
        mapInfo = torch.tensor(mapInfo, dtype = torch.long)
        #在Module中的buffer中的参数是不会被求梯度的
        self.register_buffer("mapInfo", mapInfo) 
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        #将每一个定义的卷积层都放进这个字典
        self.convs = {}
        
        #对每一个新建立的卷积层都进行注册，使其真正成为模块并且方便调用
        for i in range(self.out_channel):
            conv = nn.Conv2d(mapInfo[:, i].sum().item(), 1, kernel_size)
            convName = "conv{}".format(i)
            self.convs[convName] = conv
            self.add_module(convName, conv)

    def forward(self, x):
        #对每一个卷积层通过映射来计算卷积，结果存储在这里
        outs = []
        
        for i in range(self.out_channel):
            mapIdx = self.mapInfo[:, i].nonzero().squeeze()
            convInput = x.index_select(1, mapIdx)
            convOutput = self.convs['conv{}'.format(i)](convInput)
            outs.append(convOutput)
        return torch.cat(outs, dim = 1)


# RBF卷积层
class RBFLayer(nn.Module):
    def __init__(self, in_features=84, out_features=10, init_weight = RBF_WEIGHT):
        super(RBFLayer, self).__init__()
        if init_weight is not None:
            self.register_buffer("weight", torch.tensor(init_weight))
        else:
            self.register_buffer("weight", torch.rand(in_features, out_features))
            
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = (x - self.weight).pow(2).sum(-2)
        return x


# from tools import progress_bar
class LeNet5(nn.Module):
    # 定义网络结构
    def __init__(self, device):
        super(LeNet5, self).__init__()
        self.C1 = nn.Conv2d(1, 6, 5, padding = 2, padding_mode = 'replicate')
        self.S2 = Subsampling(6)
        self.C3 = MapConv(6, 16, 5)
        self.S4 = Subsampling(16)
        self.C5 = nn.Conv2d(16, 120, 5)
        self.F6 = nn.Linear(120, 84)
        self.Output = RBFLayer(84, 10, RBF_WEIGHT)
        self.act = nn.Tanh()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                F_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data = torch.rand(m.weight.data.size()) * 4.8 / F_in - 2.4 / F_in
            elif isinstance(m, nn.Linear):
                F_in = m.in_features
                m.weight.data = torch.rand(m.weight.data.size()) * 4.8 / F_in - 2.4 / F_in
        
        self.to(device)

    def forward(self, x):
        x = self.C1(x)
        x = 1.7159*self.act(2*self.S2(x)/3)
        x = self.C3(x)
        x = 1.7159*self.act(2*self.S4(x)/3)
        x = self.C5(x)
        
        x = x.view(-1, 120)
        x = 1.7159*self.act(2*self.F6(x)/3)
        out = self.Output(x)
        return out

class LeNetMoudle():
    def __init__(self,batch_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.net = LeNet5(self.device)
        self.epoch_num = 25
        self.loss = loss_fn
        self.optimizer = optim.SGD(self.net.parameters(), lr = 1.0e-3, momentum=0.9)
        # self.optimizer = optim.SGD(self.net.parameters(), lr = 1.0e-3)
        self.lossList = []
        self.testError = []
        self.trainError = []
        self.trainnum = 0
        self.testnum = 0

    def data(self):
        transform = transforms.Compose([
            #将图片转化为Tensor格式
            transforms.ToTensor(),
            #数据归一化处理
            transforms.Normalize((0.13066047430038452,), (0.30810782313346863,)),
        ])
        #Data set
        train_dataset = datasets.MNIST(root='../data',
                                    train=True,
                                    transform=transform,
                                    download=True)
        test_dataset = datasets.MNIST(root='../data',
                                    train=False,
                                    transform=transform)
        # Data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=False)
        self.trainnum = len(self.train_loader)
        self.testnum = len(self.test_loader)

    def train(self):
        for epoch in range(self.epoch_num):
            lossSum = 0.0
            log.info('epoch: {:02d} / {:d}'.format(epoch+1, self.epoch_num))
            # self.adjust_lr(epoch)

            for i, (image, label) in enumerate(self.train_loader):
                x = image.to(self.device)
                y = label.to(self.device)

                #梯度下降与参数更新
                out = self.net(x)
                self.optimizer.zero_grad()
                loss = self.loss(out, y)
                loss.backward()
                self.optimizer.step()

                lossSum += loss.item()
                if (i+1)%100==0:
                    log.info('{:03d}/{:d} '.format((i+1), self.trainnum
                    ) + 'lossAver:'+str(lossSum/(i+1)))
                # print(loss.item())
                # progress_bar(epoch*self.trainnum+i, self.epoch_num*self.trainnum)
                # break

            self.lossList.append(lossSum / (self.trainnum))
            log.info('lossList:'+str(self.lossList))
            self.test()
        torch.save(self.net.state_dict(), f'./module{batch_size}.pth')

    def test(self):
        with torch.no_grad():
            errorNum = 0
            for img, label in self.train_loader:
                x = img.to(self.device)
                out = self.net(x)
                _, pred_y = out.min(dim = 1)
                label = label.to(self.device)
                result = (pred_y != label).nonzero().squeeze(-1)
                errorNum += result.size()[0]
            self.trainError.append(errorNum / (self.trainnum*self.batch_size))
        log.info('trainError:'+str(self.trainError))
        #测试部分，每训练一个epoch就在测试集上进行错误率的求解与保存
        with torch.no_grad():
            errorNum = 0
            for img, label in self.test_loader:
                x = img.to(self.device)
                out = self.net(x)
                _, pred_y = out.min(dim = 1)
                label = label.to(self.device)
                result = (pred_y != label).nonzero().squeeze(-1)
                errorNum += result.size()[0]
            self.testError.append(errorNum / (self.testnum*self.batch_size))
        log.info('testError:'+str(self.testError))

    # 学习率调节
    def adjust_lr(self, epoch):
        if epoch < 5:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 1.0e-3
        elif epoch < 10:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 5.0e-4
        elif epoch < 15:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 2.0e-4
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 1.0e-4

        
def main():
    m = LeNetMoudle(batch_size)
    m.data()
    m.train()

    with open(os.path.join(os.getcwd(), f'result{batch_size}.json'),'w',encoding='utf8') as f:
        json.dump([m.lossList, m.trainError, m.testError], f, ensure_ascii=False)

if __name__ == "__main__":
    batch_size = 100
    log = Logger(f'run{batch_size}.log')
    main()

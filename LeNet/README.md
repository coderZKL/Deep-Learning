## LeNet
- 数据集

  实际训练网络使用的图片是16 x 16，并且将灰度值的范围通过变换转换到了[-1, 1]的范围内，而Pytorch提供的MNIST数据集里面是尺寸为28 x 28，像素值为[0, 255]的图片，因此在实际进行网络训练之前，我们要对数据集中的数据进行简单的处理。

- 网络结构

  ```python
  self.conv1 = nn.Conv2d(1, 12, 5, stride=2, padding=2)
  self.conv2 = nn.Conv2d(12, 12, 5, stride=2, padding=2)
  self.fc1 = nn.Linear(12*4*4, 30)
  self.fc2 = nn.Linear(30, 10)
  self.act = nn.Tanh()
  ```

- 训练

  初始化参数：-2.4F ~ 2.4F F为输入通道数

  训练方法：随机梯度下降（SGD），batch_size为1

  激活函数：1.7159Tanh(2/3 * x)

  损失函数：MSELoss
  
- 结果

  ![results](.\experiment1\results.jpg)

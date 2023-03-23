## LeNet-5
- 数据集

  实际训练网络使用的图片是28 x 28，并且将灰度值的范围通过变换转换到了[-1, 1.175]的范围内（后续做了一组归一到均值为0，方差为1的对照，归一的效果更好），我们要对数据集中的数据进行简单的处理。

- 网络结构

  ```python
  self.C1 = nn.Conv2d(1, 6, 5, padding = 2, padding_mode = 'replicate')
  self.S2 = Subsampling(6)
  self.C3 = MapConv(6, 16, 5)
  self.S4 = Subsampling(16)
  self.C5 = nn.Conv2d(16, 120, 5)
  self.F6 = nn.Linear(120, 84)
  self.Output = RBFLayer(84, 10, RBF_WEIGHT)
  self.act = nn.Tanh()
  ```
  
  - S2是自定义的平均池化层，平均后乘上权重和偏置，参数需要添加到模型中并初始化
  
  - C3是自定义的卷积层，不是在所有特征图上卷积，而是从输入中挑选一部分，具体映射关系如下图：
  
    ![1](pic\1.jpg)
  
  - Output层是一层RBF层，比较最终得到的84维特征向量与各数字对应向量之间的距离，这也是最后的loss函数（原文中为了引入类间对抗加了其他项，见相关实验）
  
- 训练

  初始化参数：-2.4F ~ 2.4F F为输入通道数

  训练方法：随机梯度下降（SGD），batch_size为1（后续实验比较了不同的size）

  激活函数：1.7159Tanh(2/3 * x)

  损失函数：Label对应的Output层输出的平均值

- 结果

  - 调节学习率与使用冲量的对比（batch_size为100）
  ![results](data_momentum\\result.jpg)
  - 不同和bantch_size情况下的对比（使用冲量）
  ![results](data_batch\\result1.jpg)
  - 使用100epoch和32batch_size的结果
  ![results](data_batch\\result2.jpg)
  最后的结果为：
    - loss：4.827073018201192
    - trainError：0.019
    - testError：0.021665335463258786
  - 改进loss函数发现结果并不理想（原文学习率的二阶调节没有复现）
  - 比较不同数据归一化方式的影响
  ![results](data_normal\\result.jpg)

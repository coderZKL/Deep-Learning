import sys
import logging
import numpy as np
import json
import matplotlib.pyplot as plt

def progress_bar(i, total):
    print("\r", end="")
    now = (i/total)*100
    print("进度: {:.2f}%: ".format(now), "▓" * (round(now) // 2), end="")
    sys.stdout.flush()


class Logger:
    def __init__(self, path, Flevel = logging.DEBUG):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        #设置文件日志
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        fh.setLevel(Flevel)
        self.logger.addHandler(fh)
     
    def debug(self,message):
        self.logger.debug(message)
    
    def info(self,message):
        self.logger.info(message)
    
    def war(self,message):
        self.logger.warn(message)
    
    def error(self,message):
        self.logger.error(message)
    
    def cri(self,message):
        self.logger.critical(message)


_zero = [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, +1, +1, +1, -1, -1] + \
        [-1, +1, +1, -1, +1, +1, -1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [-1, +1, +1, -1, +1, +1, -1] + \
        [-1, -1, +1, +1, +1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1]

_one = [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, +1, +1, +1, -1, -1] + \
       [-1, +1, +1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, +1, +1, +1, +1, +1, +1] + \
       [-1, -1, -1, -1, -1, -1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1]

_two = [-1, +1, +1, +1, +1, +1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1] + \
       [-1, +1, +1, +1, +1, +1, -1] + \
       [+1, +1, -1, -1, -1, +1, +1] + \
       [+1, -1, -1, -1, -1, +1, +1] + \
       [-1, -1, -1, -1, +1, +1, -1] + \
       [-1, -1, +1, +1, +1, -1, -1] + \
       [-1, +1, +1, -1, -1, -1, -1] + \
       [+1, +1, -1, -1, -1, -1, -1] + \
       [+1, +1, +1, +1, +1, +1, +1] + \
       [-1, -1, -1, -1, -1, -1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1]

_three = [+1, +1, +1, +1, +1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, +1, +1, -1] + \
         [-1, -1, -1, +1, +1, -1, -1] + \
         [-1, -1, +1, +1, +1, +1, -1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [-1, +1, +1, +1, +1, +1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1]

_four = [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, +1, +1, -1, -1, +1, +1] + \
        [-1, +1, +1, -1, -1, +1, +1] + \
        [+1, +1, +1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, +1, +1, +1] + \
        [-1, +1, +1, +1, +1, +1, +1] + \
        [-1, -1, -1, -1, -1, +1, +1] + \
        [-1, -1, -1, -1, -1, +1, +1]

_five = [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [+1, +1, +1, +1, +1, +1, +1] + \
        [+1, +1, -1, -1, -1, -1, -1] + \
        [+1, +1, -1, -1, -1, -1, -1] + \
        [-1, +1, +1, +1, +1, -1, -1] + \
        [-1, -1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1]

_six = [-1, -1, +1, +1, +1, +1, -1] + \
       [-1, +1, +1, -1, -1, -1, -1] + \
       [+1, +1, -1, -1, -1, -1, -1] + \
       [+1, +1, -1, -1, -1, -1, -1] + \
       [+1, +1, +1, +1, +1, +1, -1] + \
       [+1, +1, +1, -1, -1, +1, +1] + \
       [+1, +1, -1, -1, -1, +1, +1] + \
       [+1, +1, -1, -1, -1, +1, +1] + \
       [+1, +1, +1, -1, -1, +1, +1] + \
       [-1, +1, +1, +1, +1, +1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1]

_seven = [+1, +1, +1, +1, +1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, +1, +1, -1] + \
         [-1, -1, -1, +1, +1, -1, -1] + \
         [-1, -1, -1, +1, +1, -1, -1] + \
         [-1, -1, +1, +1, -1, -1, -1] + \
         [-1, -1, +1, +1, -1, -1, -1] + \
         [-1, -1, +1, +1, -1, -1, -1] + \
         [-1, -1, +1, +1, -1, -1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1]

_eight = [-1, +1, +1, +1, +1, +1, -1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [-1, +1, +1, +1, +1, +1, -1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [-1, +1, +1, +1, +1, +1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1]

_nine = [-1, +1, +1, +1, +1, +1, -1] + \
        [+1, +1, -1, -1, +1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, +1, +1, +1] + \
        [-1, +1, +1, +1, +1, +1, +1] + \
        [-1, -1, -1, -1, -1, +1, +1] + \
        [-1, -1, -1, -1, -1, +1, +1] + \
        [-1, -1, -1, -1, +1, +1, -1] + \
        [-1, +1, +1, +1, +1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1]


RBF_WEIGHT = np.array([_zero, _one, _two, _three, _four, _five, _six, _seven, _eight, _nine]).transpose()

def draw_batch():
    batch_list = [6,16,32,60]
    loss = {}
    testError = {}
    for batch in batch_list:
        path = f'./data_batch/result{batch}.json'
        with open(path,'r',encoding='utf8') as f:
            result = json.loads(f.read())

        loss_ = result[0]
        test_ = result[2]
        test_ = [num * 100 for num in test_]
        loss[str(batch)] = loss_
        testError[str(batch)] = test_

    plt.subplot(1, 2, 1)
    for batch in loss.keys():
        plt.plot(loss[batch], label=batch)
    plt.legend()
    plt.title('loss')

    plt.subplot(1, 2, 2)
    for batch in testError.keys():
        plt.plot(testError[batch], label=batch)
    plt.title('testError')
    plt.legend()
    plt.show()

def draw():
    path = f'./data_normal/result.json'
    with open(path,'r',encoding='utf8') as f:
        result1 = json.loads(f.read())
    loss1 = result1[0]
    train1 = result1[1]
    test1 = result1[2]

    path = f'./data_original/result.json'
    with open(path,'r',encoding='utf8') as f:
        result2 = json.loads(f.read())
    loss2 = result2[0]
    train2 = result2[1]
    test2 = result2[2]

    test1 = [num * 100 for num in test1]
    train1 = [num * 100 for num in train1]
    test2 = [num * 100 for num in test2]
    train2 = [num * 100 for num in train2]
    
    plt.subplot(1, 3, 1)
    plt.plot(loss1, label="normal")
    plt.plot(loss2, label="original")
    plt.title('loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train1, label="normal")
    plt.plot(train2, label="original")
    plt.title('trainError')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(test1, label="normal")
    plt.plot(test2, label="original")
    plt.title('testError')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    draw()

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


def draw():
    path = 'result.json'
    with open(path,'r',encoding='utf8') as f:
        result = json.loads(f.read())
    loss = result[0]
    test = result[1]

    test = [num * 100 for num in test]
    
    plt.subplot(1, 2, 1)
    plt.plot(loss)
    plt.title('loss')

    plt.subplot(1, 2, 2)
    plt.plot(test)
    plt.title('testError')

    plt.show()

if __name__ == "__main__":
    draw()

import json
import matplotlib.pyplot as plt

path = './experiment2/result.json'
with open(path,'r',encoding='utf8') as f:
    results = json.loads(f.read())

loss = results[0]
test = results[1]
plt.subplot(1, 2, 1)
plt.plot(loss)
plt.title('loss')
plt.subplot(1, 2, 2)
testError = [num * 100 for num in test]
plt.plot(testError)
plt.title('testError')
plt.show()

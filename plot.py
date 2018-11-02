import matplotlib.pyplot as plt
import numpy as np



testAccFile = 'Alex-64-800-L1-sgd-loss_ratio:1-1-dataAug:1.log.val.acc'

def readFile(filename):
    res = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip(' ').strip('\n')
            if(tmp=='' or tmp=='Acc:'):
              continue
            res.append(float(tmp))
            #res.append(float())
    return res


testAcc = np.array(readFile(testAccFile))






plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.plot(testAcc, '-o', label='train')



plt.gcf().set_size_inches(15, 15)
plt.savefig('figure.jpg')

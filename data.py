import pickle
import os
import matplotlib.pyplot as plt
import matplotlib

['QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

matplotlib.use('Qt')

def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

def main():
    path = '../data/cifar-10-batches-py'
    files = os.listdir(path)
    for f in files:
        if f.startswith('data'):
            d = unpickle(os.path.join(path, f))
            break

    x = d[b'data'][0]
    x = x.reshape(32,32, 3)

    plt.imshow(x)
    #plt.savefig('test.png')
    plt.show()

if __name__ == '__main__':
    main()


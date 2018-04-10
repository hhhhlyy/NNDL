########################################################################
#this project is the exercise 1 for NNDL class from Fudan University.
#author: Li Yingyue
#contact me: liyingyuelyy@outlook.com
########################################################################

import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile

import model

def main(argv=None):
    #training data spreads from -pi to pi
    # x_train = np.linspace(-np.pi, np.pi, 200).reshape(200, -1)
    x_train = np.linspace(-6, 6, 200).reshape(200, -1)

    #define the function to train here, to sample training data.
    # y_train = np.sin(x_train) +1.5 #sin
    # y_train = x_train+2           #linear
    y_train = x_train * x_train  #square

    # x_test = np.linspace(np.pi, 2*np.pi, 60).reshape(60, -1)
    x_test = np.linspace(-3, 3, 60).reshape(60, -1)

    losses = []
    #size mean the size of ReLU,
    #   e.g. size=10 means the two ReLU layers' weight matrix
    #   are 1x10 and 10x1 separately.
    #we check from 10 to 50 to find the best size to minimize the loss.
    for size in range(3, 50):
        print('current size = ', size)

        #network definition
        #model.ReLU has no trainable variables,
        #   all variables are contained in FC layer
        network = []
        network.append(model.FC(1, size))
        network.append(model.ReLU())
        network.append(model.FC(size, 1))
        network.append(model.ReLU())

        #training procedure.
        #number of iteration should be adjusted
        #   according to different function to fit.
        loss_buffer = []
        for i in range(20000):
            loss = model.train(network, x_train, y_train)
            loss_buffer.append(loss)
        print(np.mean(loss_buffer[-1000:]))
        losses.append(np.mean(loss_buffer[-1000:]))

        #drawing figures
        fig_loss = plt.figure()
        ax_loss = fig_loss.add_subplot(111)
        p_loss = ax_loss.plot(np.linspace(0,10000,10000), loss_buffer[1:10001], '.')
        ax_loss.set_xlabel('x-points')
        ax_loss.set_ylabel('y-points')
        name_loss = 'curves/square_loss_%d' % size
        fig_loss.savefig(name_loss)

        fig_result = plt.figure()
        ax = fig_result.add_subplot(111)
        p = ax.plot(x_test, model.inference(network, x_test), '*')
        ax.set_xlabel('x-points')
        ax.set_ylabel('y-points')
        name = 'curves/square_result_%d' % size
        fig_result.savefig(name)

        plt.close('all')

    #picking up the best result
    loss_min = min(losses)
    best_h = losses.index(loss_min)+3
    copyfile('curves/square_result_%d.png' %best_h, 'best_curves/square_result_%d.png' %best_h)
    copyfile('curves/square_loss_%d.png' %best_h, 'best_curves/square_loss_%d.png' %best_h)

if __name__ == '__main__':
    main()
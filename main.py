import chainer
import chainer.links as L
import chainer.functions as F
import chainer.cuda
import argparse
import os
import cv2
import numpy as np
import glob
import config as cf
from data_loader import DataLoader
from network import Mynet

class main():

    def __init__(self):
        pass

    def train(self):

        #ネットワークのロード
        #Network loading
        model=Mynet()

        ##opimizer
        optimizer=chainer.optimizers.MomentumSGD(cf.Learning_rate)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

        #データのロード
        #Data Loading
        d1=DataLoader(phase='Train',shuffle=True)
        d1_test=DataLoader(phase='Test',shuffle=True)
        test_imgs,test_gts=d1_test.get_minibatch(shuffle=False)

        test_imgs=chainer.Variable(test_imgs)
        test_gts=chainer.Variable(test_gts)

        #学習開始
        #Start Learning
        print('Epoch, Loss, Accu, Loss-test, Accu-test')
        train_losses=[]
        train_accuracies=[]

        for epoch in range(cf.Step):
            epoch+=1
            imgs,gts=d1.get_minibatch(shuffle=True)

            x=chainer.Variable(imgs)
            t=chainer.Variable(gts)
            x=imgs
            t=gts

            y=model(x)

            loss=F.softmax_cross_entropy(y,t)
            accuracy=F.accuracy(y,t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            train_losses.append(loss.data)
            train_accuracies.append(accuracy.data)

            if epoch % 10==0:

                y_test=model(test_imgs)
                loss_test=F.softmax_cross_entropy(y_test,test_gts)
                accu_test=F.accuracy(y_test,test_gts)

                loss_test=loss_test.data
                accu_test=accu_test.data

                print('{:3d}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(epoch,np.mean(train_losses),np.mean(train_accuracies),np.mean(loss_test),np.mean(accu_test)))

            os.makedirs(cf.Save_dir,exist_ok=True)
            chainer.serializers.save_npz(cf.Save_path,model)

if __name__=='__main__':
    main=main()
    main.train()

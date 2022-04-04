import chainer
import chainer.links as L
import chainer.functions as F
import config as cf
class Mynet(chainer.Chain):
    def __init__(self):
        super(Mynet,self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None,32,ksize=3,pad=1)
#入力チャネル、出力チャネル、フィルタサイズ、パディング幅（画像の縁の外側に仮想的な画素を設けること、フィルタサイズ /2 の切捨て数）
#Input channel, output channel, filter size, padding width(virtual pixels outside image edges, truncation number of filter size/2)
            self.conv2=L.Convolution2D(None,64,ksize=3,pad =1)
            self.conv3=L.Convolution2D(None,128,ksize=3,pad=1)
            self.fc1=L.Linear(None,256)
            self.fc_out=L.Linear(None,cf.Class_num)

    def __call__(self,x):
        conv1=self.conv1(x)
        conv1_a=F.relu(conv1)
        pool1=F.max_pooling_2d(conv1_a,ksize=2,stride=2)
        conv2=self.conv2(pool1)
        conv2_a=F.relu(conv2)
        pool2=F.max_pooling_2d(conv2_a,ksize=2,stride=2)
        # d=F.dropout(pool2,ratio=0.5)
        conv3=self.conv3(pool2)
        conv3_a=F.relu(conv3)
        pool3=F.max_pooling_2d(conv3_a,ksize=2,stride=2)
        # d=F.dropout(pool3,ratio=0.5)
        fc1=self.fc1(pool3)
        fc1_a=F.relu(fc1)
        fc1_d=F.dropout(fc1_a,ratio=0.5)
        fc_out= self.fc_out(fc1_d) #fc1_a to fc1_d
        return fc_out
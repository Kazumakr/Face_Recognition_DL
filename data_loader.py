import os
import glob
import cv2
import numpy as np
import config as cf
class DataLoader():
    def __init__(self,phase='Train',shuffle=True):
        self.datas=[]
        self.last_mb=0
        self.phase=phase
        self.gt_count=[0 for _ in range(cf.Class_num)]
        self.prepare_datas(shuffle=shuffle)
    
    def get_gt(self,img_name):
        for ind,cls in enumerate(cf.Class_label):
            if cls in img_name:
                return ind

        raise Exception("Class label Error {}".format(img_name))

    def load_image(self,img_name,h_flip=False,v_flip=False,nff_rotate=False,pff_rotate=False):
        #Image load
        img=cv2.imread (img_name)
        if img is None:
            raise Exception('file not found: {}',format(img_name))
        
        img=cv2.resize(img,(cf.Width,cf.Height))
        img=img[:,:,(2,1,0)]
        img=img/255
        ##Horizontal_flip 左右反転
        if h_flip:
            img=img[:,:: -1,:]
        ##Vertical_flip 上下反転
        if v_flip:
            img=img[:: -1,:,:]

        ##+-45digree rotate 45 度回転
        if nff_rotate:
            size=tuple([img.shape[1],img.shape[0]])
            center=tuple([int(size[0]/ 2),int(size[1]/2)])
            angle= -45.0 #ここを変えて複製してもいい, You can change this parameter and duplicate more
            scale=1.0
            rotation_mat=cv2.getRotationMatrix2D(center,angle,scale)
            img=cv2.warpAffine(img,rotation,size,flags=cv2.INTER_CUBIC)

        if pff_rotate:
            size=tuple([img.shape[1],img.shape[0]])
            center=tuple([int(size[0]/2),int(size[1]/2)])
            angle=45.0
            scale=1.0
            rotation_mat=cv2.getRotationMatrix2D(center,angle,scale)
            img=cv2.warpAffine(img,rotation,size,flags=cv2.INTER_CUBIC)

        img=img.transpose(2,0,1)
        return img

    def data_augmentation(self,h_flip=False,v_flip=False,nff_rotate=False,pff_rotate=False):
        print()
        print(' || -*- Data Augmentation-*-')
        if h_flip:
            self.add_horizontal_flip()
            print(' || Added horizontal flip')
        if v_flip:
            self.add_vertical_flip()
            print(' || Added vertical flip')
        if nff_rotate:
            self.add_nff_rotate()
            print(' || Added -45 rotation')
        if pff_rotate:
            self.add_pff_rotate()
            print(' || Added 45 rotation')
            print('\n')
            print('\n')
    
    def add_horizontal_flip(self):

        ##Add Horizontal flipped image data

        new_data=[]

        for data in self.datas:
            _data={'img_path':data['img_path'],
                    'gt_path':data['gt_path'],
                    'h_flip':True,
                    'v_flip':data['v_flip'],
                    'nff_rotate':data['nff_rotate'],
                    'pff_rotate':data['pff_rotate']
            }

            new_data.append(_data)
            gt=self.get_gt(data['img_path'])
            self.gt_count[gt]+=1

        self.datas.extend(new_data)
    
    def add_vertical_flip(self):

        ##Add Horizontal flipped image data
        new_data=[]

        for data in self.datas:
            _data={'img_path':data['img_path'],
                    'gt_path':data['gt_path'],
                    'h_flip':data['h_flip'],
                    'v_flip':True,
                    'nff_rotate':data['nff_rotate'],
                    'pff_rotate':data['pff_rotate']
            }

            new_data.append(_data)
            gt=self.get_gt(data['img_path'])
            self.gt_count[gt]+=1

        self.datas.extend(new_data)

    def add_nff_rotate(self):
        new_data=[]

        for data in self.datas:
            _data={'img_path':data['img_path'],
                    'gt_path':data['gt_path'],
                    'h_flip':data['h_flip'],
                    'v_flip':data['v_flip'],
                    'nff_rotate':True,
                    'pff_rotate':data['pff_rotate']
            }

            new_data.append(_data)
            gt=self.get_gt(data['img_path'])
            self.gt_count[gt]+=1

        self.datas.extend(new_data)

    def add_pff_rotate(self):

        new_data=[]

        for data in self.datas:
            _data={'img_path':data['img_path'],
                    'gt_path':data['gt_path'],
                    'h_flip':data['h_flip'],
                    'v_flip':data['v_flip'],
                    'nff_rotate':data['nff_rotate'],
                    'pff_rotate':True
            }

            new_data.append(_data)
            gt=self.get_gt(data['img_path'])
            self.gt_count[gt]+=1

        self.datas.extend(new_data)
    
    def display_gt_statistic(self):

        print()
        print(' -*- Training label statistic -*-')
        self.display_data_total()

        for i,gt in enumerate(self.gt_count):
            print(' -{}:{}'.format(cf.Class_label[i],gt))

    def set_index(self,shuffle=True):
        self.data_n=len(self.datas)

        self.indices=np.arange(self.data_n)

        if shuffle:
            np.random.seed(0)
            np.random.shuffle(self.indices)

    def prepare_datas(self,shuffle=True):

        if self.phase=='Train':
            dir_paths=cf.Train_dirs
        elif self.phase=='Test':
            dir_paths=cf.Test_dirs

        print()
        print('--------------')
        print('Data Load (phase: {})'.format(self.phase))

        for dir_path in dir_paths:
            files=glob.glob(dir_path + '/*.jpg')

            load_count=0
            for img_path in files:
                gt=self.get_gt(img_path)

                img=self.load_image(img_path)
                gt_path=1

                data={'img_path': img_path,
                    'gt_path': gt_path,
                    'h_flip': False,
                    'v_flip': False,
                    'nff_rotate':False,
                    'pff_rotate':False
                }

                self.datas.append(data)

                self.gt_count[gt]+=1
                load_count+=1

            print(' - {} - {}datas -> loaded {}'.format(dir_path,len(files),load_count))

        self.display_gt_statistic()

        self.check_data_num()
        if self.phase=='Train':
            self.data_augmentation(h_flip=cf.Horizontal_flip,v_flip=cf.Vertical_flip,nff_rotate=cf.nff_rotate,pff_rotate=cf.pff_rotate)

        self.display_gt_statistic()

        self.set_index(shuffle=shuffle)

    def check_data_num(self):
        if self.phase=='Train' and len(self.datas) < 1:
            raise Exception("Training data not found!")

    def display_data_total(self):
        print(' Total data: {}'.format(len(self.datas)))

#ミニバッチ　1エポック毎にバッチサイズが加算され、データ数を上回るときバッチサイズからデータ数が引かれる。バッチサイズは小さいとパラメータの更新が増え、1つのデータに敏感に反応し、大きいと学習が早くなる。学習率に応じて設定する必要があるため、このようにバッチサイズを変更させている。参考文献等ではバッチサイズは固定している。
#Mini-batch The batch size is added for each epoch, and the number of data is subtracted from the batch size when it exceeds the number of data. Smaller batch sizes increase parameter updates and are more sensitive to a single data point, while larger batch sizes result in faster learning. The batch size is changed in this way because it must be set according to the learning rate. In the references and other literature, the batch size is fixed.

    def get_minibatch_index(self,shuffle=False):

        if self.phase=='Train':
            mb=cf.Minibatch
        elif self.phase=='Test':
            mb=cf.Minibatch

        _last=self.last_mb+mb
        
        if _last >= self.data_n:
            mb_inds=self.indices[self.last_mb:]
            self.last_mb=_last - self.data_n
        
            if shuffle:
                np.random.seed(0)
                np.random.shuffle(self.indices)

            _mb_inds=self.indices[:self.last_mb]
            mb_inds=np.hstack((mb_inds,mb_inds))

        else:
            mb_inds=self.indices[self.last_mb : self.last_mb+mb]
            self.last_mb += mb

        self.mb_inds=mb_inds

    def get_minibatch(self,shuffle=True):

        if self.phase=='Train':
            mb=cf.Minibatch
        elif self.phase=='Test':
            mb=cf.Minibatch

        self.get_minibatch_index(shuffle=True)

        imgs=np.zeros((mb,3,cf.Height,cf.Width),dtype=np.float32)

        gts=np.zeros((mb),dtype=np.int32)

        for i,ind in enumerate(self.mb_inds):
            data=self.datas[ind]
            img=self.load_image(data['img_path'],h_flip=data['h_flip'])
            gt=self.get_gt(data['img_path'])

            imgs[i]=img
            gts[i]=gt

        return imgs,gts

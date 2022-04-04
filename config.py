import os
#画像の入力サイズ設定
#Set image input size
Height=64
Width=64
#クラスラベル設定（ファイル名の頭で判別）
#Class label setting (identified at the beginning of the file name)
Class_label=[
    'Kuramoto',
    'Sawai',
    'Sone',
    'Oyama',
    'side',
    'Kurihara',
]
Class_num=len(Class_label)
#パスを通す
#Path
Train_dirs=[
    './Data/train/Kuramoto',
    './Data/train/Sawai',
    './Data/train/Sone'
    './Data/train/Oyama',
    './Data/train/side'
    './Data/train/Kurihara',
]
Test_dirs=[
    './Data/test/Kuramoto',
    './Data/test/Sawai',
    './Data/test/Sone'
    './Data/test/Oyama',
    './Data/test/side'
    './Data/test/Kurihara',
]
##学習の各種パラメータ設定
#Various parameter settings for learning
Step=600
Minibatch=64
Learning_rate=0.0001
##前処理の各種設定
#Various pre-processing settings
Horizontal_flip=True
Vertical_flip=True
nff_rotate=True
pff_rotate=True
##学習モデルの保存先設定
#Setting the destination for the training model
Save_dir='out'
Model_name='CNN.npz'
Save_path=os.path.join(Save_dir,Model_name)
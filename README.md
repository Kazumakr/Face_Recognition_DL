# Face_Recognition_DL

Face Recognition Using Deep Learning

This was my senior project in 2019.

## Table of Contents (Optional)

- [Description](#description)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)
- [References](#references)

## Description

In recent years, facial recognition technology has been used at airports and entertainment facilities.
The current situation is that user must face the front and stand still. For face recognition at airports, the face image in the IC chip of the passport is matched with the face image taken by the camera at the authentication gate to confirm the identity of the customer.
The system is based on the principle of "check-in". Accurate authentication is not possible for moving objects.
Therefore, in this project, I used deep learning to develop a real-time, highly accurate face. The purpose of this project is to verify face recognition using a USB web camera.

### Built With

- [MongoDB](https://www.mongodb.com/)
- [Express.js](https://expressjs.com/)
- [Node.js](https://nodejs.org/)

## Features

- Create training data
- Learning and Testing

## Requirements

- OS windows7
- CPU Intel Core i7-4770 @3.40GHz
- Memory 16GB
- Python 3.7.0
- Anaconda 3
- Chaniner 5.0.0
- Camera Logicool HD Pro Webcam c920r

## Usage

Create two folders, train and test, in the Data folder, and prepare files for each class. Execute get_front.py, main.py, and test_camera.py in this order.
Photos of lab members, which are data, are not published due to privacy issues.

## License

License under the [MIT License](LICENSE)

## References

- [HELLO CYBERNETICS, 誤差逆伝播法（バックプロパゲーション）とは](https//www.hellocybernetics.tech/entry/2017/02/23/213120)
- [AISIA, 畳み込みニューラルネットワーク \_CNN Vol.16](https://products.sint.co.jp/aisia/blog/vol1-16)
- 新納浩幸 Chainer v2 による実践深層学習, オーム社 , 第 1 版第 2 刷 2017 年
- 石川聡彦 Python で動かして学ぶ！あたらしい深層学習の教科書, 翔泳社 初版第 1 刷, 2018 年
- 牧野浩二, 西崎博光, Python による深層強化学習入門, オーム社, 第 1 版第 1 刷, 2018 年
- 斎藤康毅 ゼロから作る Deep Learning Python で学ぶディープラーニングの理論と実装 , オライリージャパン , 初版第 11 刷 , 2018
- 岡谷貴之 , 機械学習プロフェッショナルシリーズ 深層学習 , 講談社 , 第 5 刷 , 2015
- 原田達也 , 機械学習プロフェッショナルシリーズ 画像認識 , 講談社 , 第 4 刷 , 2018
- 中部大学工学部情報工学科 藤吉研究室 , 局所特徴量と統計学習手法による物体検出

#! /bin/bash

cd dataset
wget http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
mkdir bin
unzip train2014.zip
tar -czvf 256_ObjectCategories.tar caltech256
tar -czvf 101_ObjectCategories.tar.gz caltech101

mkdir caltech101/train
mkdir caltech101/val
mv caltech101/ caltech101/train
mkdir caltech256/train
mkdir caltech256/val
mv caltech256/ caltech256/train
#! /bin/bash
mkdir dataset
cd dataset
wget http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
mkdir bin
unzip train2014.zip

tar xvf 256_ObjectCategories.tar 
rm 256_ObjectCategories.tar
mkdir -p caltech256/train
mkdir caltech256/val
mv 256_ObjectCategories/* caltech256/train/
rm -rf 256_ObjectCategories/

tar xvf 101_ObjectCategories.tar.gz
rm 101_ObjectCategories.tar.gz
mkdir -p caltech101/train
mkdir caltech101/val
mv 101_ObjectCategories/* caltech101/train/
rm -rf 101_ObjectCategories/
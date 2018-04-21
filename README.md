# Neural Transfer for Image Data Augmentation

This project is to use Neural Transfer as a Image Data Augmentation strategy. There are mainly two parts, using neural networks to generate images and using vgg network to evaluate the results.

### Implementation Details
The implementation is based on TensorFlow and Keras. The base code for style transfer is from https://github.com/lengstrom/fast-style-transfer. I just deleted some useless files and modified the reading files code to support the clatech folder strucure.

###Setup
- git clone git@gitlab.scss.tcd.ie:xzheng/dissertation.git
- cd dissertation
- ./setup.sh

### Training Style Transfer Networks
Use `style.py` to train a new style transfer network. Run `python style.py` to view all the possible parameters. Remeber to create the checkpoint folder before runing below command. Example usage:

    python style.py --style styles/rain_princess.jpg \
      --checkpoint-dir checkpoint/sunflower/ \
      --content-weight 1.5e1 \
      --checkpoint-iterations 100 \
      --batch-size 10 \

### Generate Images with Style
Use `evaluate.py` to generate images. Run `python evaluate.py` to view all the possible parameters. Example usage:

    python evaluate.py --checkpoint style_models/rain_princess.ckpt \
      --in-path /home/zen/dissertation/dataset/caltech101/train \
      --out-path /home/zen/dissertation/dataset/caltech101/train \
      --allow-different-dimensions

### Evaluate with VGG16/VGG19
Before evaluation, we need to split the original dataset into training set and testing set. By running below command, the raw dataset will be 7/3 splited.

    python split_caltech.py

Use `train.py` to evaluate effect of augmentated images. Run `python train.py` will run the default classification training with VGG16 and Caltech101. You can provide prameters to specific the models and the dataset.

    python train.py --dataset caltech256 \
      --model vgg19 \
      --style wave
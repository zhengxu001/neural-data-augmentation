import os
import scipy.io
import scipy.misc
import config
import numpy as np
import argparse
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
K.set_image_dim_ordering('th')

def augmentation(aug_strategy):
    if aug_strategy=="NA":
        print("No Traditional ImageAugmentation\n")
        train_datagen = ImageDataGenerator(
            rescale=1. / 255)
        val_datagen = ImageDataGenerator(
            rescale=1. / 255)
    elif aug_strategy=="filp":
        print("Use Filp as Traditional ImageAugmentation\n")
        train_datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True)
        val_datagen = ImageDataGenerator(
            rescale=1. / 255)
    else:
        print("Please choose the right ImageAugmentation Strategy\n")

    return train_datagen, val_datagen

def get_data(dataset, style):
    if dataset == "caltech101":
        if style == "wave"
            train_dataset = config.CAL101_TRAIN_WAVE
            validation_dataset = config.CAL101_VAL_WAVE
            class_number = 102
        else:
            train_dataset = config.CAL101_TRAIN
            validation_dataset = config.CAL101_VAL
            class_number = 102
    elif dataset == "caltech256":
        train_dataset = config.CAL256_TRAIN
        validation_dataset = config.CAL256_VAL
        class_number = 257
    else:
        print("No Such Dataset Supported! Please specify the correct Dataset Name")

    return train_dataset, validation_dataset, class_number

def build_vgg_models(model, dataset, epochs, aug_strategy, style):
    train_dataset, validation_dataset, class_number = get_data(dataset, style)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    tensorboard_dir = os.path.join(config.TENSOR_BOARD, model+"_"+dataset+"_"+aug_strategy+"_"+style)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tbCallBack = TensorBoard(log_dir= tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)
    vgg = eval(model)(weights='imagenet')
    fc2 = vgg.get_layer('fc2').output
    prediction = Dense(output_dim=class_number, activation='softmax', name='logit')(fc2)
    model = Model(input=vgg.input, output=prediction)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0001), metrics=['accuracy'])
    train_datagen, val_datagen = augmentation(aug_strategy)
    train_generator = train_datagen.flow_from_directory(
        train_dataset,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')
    val_generator = train_datagen.flow_from_directory(
        validation_dataset,
        target_size=(224,224),
        batch_size=32,
        shuffle=False, 
        class_mode='categorical')
    model.fit_generator(train_generator, validation_data=val_generator,
          epochs=epochs, steps_per_epoch = None, callbacks=[earlyStopping, tbCallBack])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='caltech101',
                        help='caltech101 or caltech256')
    parser.add_argument('--num_epochs', type=int,
                        default=40,
                        help='number of epochs.')
    parser.add_argument('--aug_strategy', type=str,
                        default='NA',
                        help='Traditional Data Argument Strategy.')
    parser.add_argument('--model', type=str,
                        default='vgg16',
                        help='model name')
    parser.add_argument('--style', type=str,
                        default="NA",
                        help='choose the neural style for the image data.')
    args, unparsed = parser.parse_known_args()
    build_vgg_models(args.model.upper(), args.dataset, args.num_epochs, args.aug_strategy, args.style)
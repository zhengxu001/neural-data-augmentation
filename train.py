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
K.set_image_dim_ordering('th')
valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        img = np.transpose(np.array([img, img, img]), (2, 0, 1))
    return img

def augmentation(aug_strategy):
    if aug_strategy=="NA":
        print("No Traditional ImageAugmentation\n")
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True)
    elif aug_strategy=="filp":
        print("Use Filp as Traditional ImageAugmentation\n")
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            horizontal_flip=True,
            vertical_flip=True)
    else:
        print("Please choose the right ImageAugmentation Strategy\n")

    return datagen

def load_data_according_to_style(style):
    if style == 0:
        X_train, y_train = get_data(config.CAL101_TRAIN)
        X_test, y_test = get_data(config.CAL101_TEST)
    elif style == 1:
        X_train, y_train = get_data(config.CAL101_TRAIN_STYLE1)
        X_test, y_test = get_data(config.CAL101_TEST_STYLE1)
    else:
        print("No Such Style")
    return X_train, y_train, X_test, y_test

def get_data(path):
    print ("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))
    categories = sorted(os.listdir(path))
    X = []; y = []
    for i, category in enumerate(categories):
        for f in os.listdir(path + "/" + category):
            if os.path.splitext(f)[1].lower() not in valid_exts:
                continue
            fullpath = os.path.join(path + "/" + category, f)
            img = scipy.misc.imresize(imread(fullpath), [224,224, 3])
            img = img.astype('float32')
            print(i)
            print(category)
            X.append(img)
            y.append(i)
    X = np.stack(X, axis=0)
    X = X.transpose(0, 3, 1, 2)
    y = np.stack(y, axis=0)
    y = np_utils.to_categorical(y)

def build_vgg_models(model, X_train, y_train, X_test, y_test, aug_strategy, epochs, name):
    from keras.callbacks import EarlyStopping
    from keras.callbacks import TensorBoard
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    tensorboard_dir = os.path.join(config.TENSOR_BOARD, name)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tbCallBack = TensorBoard(log_dir= tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)
    vgg = eval(model)(weights='imagenet')
    fc2 = vgg.get_layer('fc2').output
    num_classes = y_test.shape[1]
    prediction = Dense(output_dim=num_classes, activation='softmax', name='logit')(fc2)
    model = Model(input=vgg.input, output=prediction)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0001), metrics=['accuracy'])
    datagen = augmentation(aug_strategy)
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), 
          epochs=epochs, batch_size=56, shuffle=True, callbacks=[earlyStopping, tbCallBack])
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

def vgg16(X_train, y_train, X_test, y_test, aug_strategy, epochs, name):
    build_vgg_models("VGG16", X_train, y_train, X_test, y_test, aug_strategy, epochs, name)

def vgg19(X_train, y_train, X_test, y_test, aug_strategy, epochs, name):
    build_vgg_models("VGG19", X_train, y_train, X_test, y_test, aug_strategy, epochs, name)

def main(name, epochs, aug_strategy, model, style):
    X_train, y_train, X_test, y_test = load_data_according_to_style(style)
    eval(model)(model, X_train, y_train, X_test, y_test, aug_strategy, epochs, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,
                        default='default',
                        help='Name of this training run.')
    parser.add_argument('--num_epochs', type=int,
                        default=20,
                        help='number of epochs.')
    parser.add_argument('--aug_strategy', type=str,
                        default='NA',
                        help='Traditional Data Argument Strategy.')
    parser.add_argument('--model', type=str,
                        default='vgg16',
                        help='model name')
    parser.add_argument('--style', type=int,
                        default=0,
                        help='choose the neural style for the image data.')
    args, unparsed = parser.parse_known_args()
    main(args.name, args.num_epochs, args.aug_strategy, args.model, args.style)
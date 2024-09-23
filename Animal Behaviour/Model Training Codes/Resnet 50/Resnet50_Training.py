# NOTE:-
# 1. Make sure that all the libraries that are needed to run the program/train the model are
# installed/imported properly in your system. If so NO install it using pip install command
# 2. In this code i have Given the directory according to my local machine, so feel free to
# modify the directory according to where your dataset is present

import cv2
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
from keras.models import Sequential, Model,load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
# from google.colab.patches import cv2_imshow
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
from keras.preprocessing import image
from keras.initializers import glorot_uniform
import matplotlib.pyplot as plt

train_path = r"C:/Users/ksvib/Downloads/Mini Project - 7th Sem/Dataset/dataset_new/training_set"
test_path = r"C:/Users/ksvib/Downloads/Mini Project - 7th Sem/Dataset/dataset_new/testing_set"

class_names=os.listdir(train_path)
class_names_test=os.listdir(test_path)

# Printing the name of the classes and test path
print(class_names)
print(class_names_test)

#-------------------------------------------------------------------------------------------------------
# IF YOU ARE IN NEED OF PREVIEWING USE THIS CODE OR COMMENT THIS THING OUT
# Read and display the dog image
image_dog = cv2.imread(r"C:/Users/ksvib/Downloads/Mini Project - 7th Sem/Dataset/dataset_new/testing_set/dogs/dog.9997.jpg")
image_dog_rgb = cv2.cvtColor(image_dog, cv2.COLOR_BGR2RGB)
plt.imshow(image_dog_rgb)
plt.axis('off')  # Turn off axis labels
plt.title('Dog Image')
plt.show()

# Read and display the cat image
image_cat = cv2.imread(r"C:/Users/ksvib/Downloads/Mini Project - 7th Sem/Dataset/dataset_new/testing_set/cats/cat.9922.jpg")
image_cat_rgb = cv2.cvtColor(image_cat, cv2.COLOR_BGR2RGB)
plt.imshow(image_cat_rgb)
plt.axis('off')  # Turn off axis labels
plt.title('Cat Image')
plt.show()
#----------------------------------------------------------------------------------------------------------

train_datagen = ImageDataGenerator(zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15)
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(train_path,target_size=(224, 224),batch_size=32,shuffle=True,class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_path,target_size=(224,224),batch_size=32,shuffle=False,class_mode='binary')


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])  # SKIP Connection
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape=(224, 224, 3)):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

base_model = ResNet50(input_shape=(224, 224, 3))

headModel = base_model.output
headModel = Flatten()(headModel)
headModel=Dense(256, activation='relu', name='fc1',kernel_initializer=glorot_uniform(seed=0))(headModel)
headModel=Dense(128, activation='relu', name='fc2',kernel_initializer=glorot_uniform(seed=0))(headModel)
headModel = Dense( 1,activation='sigmoid', name='fc3',kernel_initializer=glorot_uniform(seed=0))(headModel)

model = Model(inputs=base_model.input, outputs=headModel)

model.summary()

base_model.load_weights(r"C:/Users/ksvib/Downloads/Mini Project - 7th Sem/Practical Implementations/ResNet 50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

for layer in base_model.layers:
    layer.trainable = False

for layer in model.layers:
    print(layer, layer.trainable)

opt = SGD(lr=1e-3, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)

mc = ModelCheckpoint(r"C:/Users/ksvib/Downloads/Mini Project - 7th Sem/Practical Implementations/ResNet 50/best_model.h5", monitor='val_accuracy', mode='max', save_best_only=True)

H = model.fit_generator(train_generator,validation_data=test_generator,epochs=100,verbose=1,callbacks=[mc,es])

# H.save("resnet_50.h5")
model.load_weights(r"C:/Users/ksvib/Downloads/Mini Project - 7th Sem/Practical Implementations/ResNet 50/best_model.h5")

#Evaluating the model on test datasets
model.evaluate(test_generator)

model_json = model.to_json()
with open(r"C:/Users/ksvib/Downloads/Mini Project - 7th Sem/Practical Implementations/ResNet 50/model.json","w") as json_file:
  json_file.write(model_json)




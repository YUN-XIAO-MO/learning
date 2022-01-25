import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score
from tensorflow.python.keras.utils import np_utils


import pandas as pd
import numpy as np



def split_data(X, y, test_data_size):
    """
    将数据分解为测试和训练数据集。

    输入
        X:数组的NumPy数组
        y:熊猫系列，这是输入数组X的标签
        Test_data_size:测试/列分割的大小。取值范围为0 ~ 1

    输出
        四个数组:X_train、X_test、y_train和y_test
    """
    return train_test_split(X, y, test_size=test_data_size, random_state=42)

def reshape_data(arr, img_rows, img_cols, channels):
    """
    将数据重塑为CNN的格式。

    输入
        arr: NumPy数组的数组。
        img_rows:图像的高度
        img_cols:图像宽度
        channels:指定图像是灰度(1)还是RGB (3)

    输出
        NumPy数组的重塑数组。
    """
    return arr.reshape(arr.shape[0], img_rows, img_cols, channels)


def cnn_model(X_train, y_train,num_filters,kernel_size,img_rows, img_cols, channels,num_classes,batch_size,num_epoch):
    model = Sequential()

    model.add(Conv2D(num_filters, (kernel_size[0], kernel_size[1]),
                     padding='valid',
                     strides=1,
                     input_shape=(img_rows, img_cols, channels), activation="relu"))

    model.add(Conv2D(num_filters, (kernel_size[0], kernel_size[1]), activation="relu"))

    model.add(Conv2D(num_filters, (kernel_size[0], kernel_size[1]), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    print("Model flattened out to: ", model.output_shape)

    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    #model = multi_gpu_model(model, gpus=nb_gpus)
    #delete 

    model.compile(loss='binary_crossentropy',  #损失函数
                  optimizer='adam',    #优化器
                  metrics=['accuracy'])   #准确率
    #model.compile方法用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准

    stop = EarlyStopping(monitor='val_acc',
                         min_delta=0.001,
                         patience=2,
                         verbose=0,
                         mode='auto')

    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch,
              verbose=1,
              validation_split=0.2,
              class_weight='auto',
              callbacks=[stop, tensor_board])

    return model

def save_model(model, score, model_name):
    """
    根据precision_score将Keras模型保存到h5文件

    输入
        model:要保存的Keras模型对象
        scpre:决定模型是否需要保存的分数。
        Model_name:需要保存的型号名
    """

    if score >= 0.75:
        print("Saving Model")
        model.save("../models/" + model_name + "_recall_" + str(round(score, 4)) + ".h5")
    else:
        print("Model Not Saved.  Score: ", score)


if __name__ == '__main__':
    # Specify parameters before model is run.
    batch_size = 16
    num_classes = 2
    num_epoch = 30

    img_rows, img_cols = 256, 256
    channels = 3
    num_filters = 32
    kernel_size = (8, 8)

    # Import data
    labels = pd.read_csv("../labels/trainLabels_master_256_v2.csv")
    # X = np.load("../data/X_train_256_v2.npy")
    X = np.load("../data/X_train.npy")
    y = np.array([1 if l >= 1 else 0 for l in labels['level']])
    # y = np.array(labels['level'])

    print("Splitting data into test/ train datasets")
    X_train, X_test, y_train, y_test = split_data(X, y, 0.2)

    print("Reshaping Data")
    X_train = reshape_data(X_train, img_rows, img_cols, channels)
    X_test = reshape_data(X_test, img_rows, img_cols, channels)

    print("X_train Shape: ", X_train.shape)
    print("X_test Shape: ", X_test.shape)

    input_shape = (img_rows, img_cols, channels)

    print("Normalizing Data")
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    print("y_train Shape: ", y_train.shape)
    print("y_test Shape: ", y_test.shape)

    print("Training Model")
    
#     model = cnn_model(X_train, y_train, kernel_size, nb_filters, channels, nb_epoch, batch_size,
#                       nb_classes, nb_gpus=8)
       #delete
    
    model = cnn_model(X_train=X_train,
                      y_train=y_train,
                      num_filters=num_filters,
                      kernel_size=kernel_size,
                      img_rows=img_rows,
                      img_cols=img_cols,
                      channels=channels,
                      num_classes=num_classes,
                      batch_size=batch_size,
                      num_epoch=num_epoch)

    print("Predicting")
    y_pred = model.predict(X_test)

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Precision: ", precision)
    print("Recall: ", recall)

    save_model(model=model, score=recall, model_name="DR_Two_Classes")
    print("Completed")

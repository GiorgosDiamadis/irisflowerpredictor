import joblib

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def create_model():
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=32, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=128, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=3, activation='softmax'))

    ann.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    xtrain, xtest, ytrain, ytest = data_preprocessing()
    ann.fit(xtrain, ytrain, batch_size=64, epochs=100, validation_data=(xtest, ytest))

    ann.save("network.h5")
    ann.save_weights("weights.h5")
    model_json = ann.to_json()
    with open("Model/model_json.json", "w") as json_file:
        json_file.write(model_json)


def data_preprocessing():
    dataset = pd.read_csv("Data/iris.csv")

    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')

    Y = Y.reshape(-1, 1)
    Y = np.array(ct.fit_transform(Y))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    joblib.dump(sc, "Model/scaler.sc")

    return x_train, x_test, y_train, y_test


create_model()

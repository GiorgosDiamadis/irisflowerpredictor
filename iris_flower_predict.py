import joblib

import tensorflow as tf
import tkinter as tk

from tkinter import messagebox

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def predict(septal_length, septal_width, petal_length, petal_width, nn):
    pred = [[]]
    pred[0].append(float(septal_length.get()))
    pred[0].append(float(septal_width.get()))
    pred[0].append(float(petal_length.get()))
    pred[0].append(float(petal_width.get()))

    scaler = joblib.load("Model/scaler.sc")

    prediction = nn.predict(scaler.transform(pred))

    maximum = -1
    for i in range(3):
        if prediction[0][i] > prediction[0][maximum]:
            maximum = i

    messagebox.showinfo("Prediction", species[maximum])

    septal_length.delete(0, 'end')
    septal_width.delete(0, 'end')
    petal_length.delete(0, 'end')
    petal_width.delete(0, 'end')


def show_gui():
    master = tk.Tk()
    master.title("Iris Flower Predictor")

    tk.Label(master, text="Septal Length").grid(row=0)
    tk.Label(master, text="Septal Width").grid(row=1)
    tk.Label(master, text="Petal Length").grid(row=2)
    tk.Label(master, text="Petal Width").grid(row=3)

    septal_length = tk.Entry(master)
    septal_width = tk.Entry(master)
    petal_length = tk.Entry(master)
    petal_width = tk.Entry(master)

    septal_length.grid(row=0, column=1)
    septal_width.grid(row=1, column=1)
    petal_length.grid(row=2, column=1)
    petal_width.grid(row=3, column=1)

    tk.Button(master, text='Predict',
              command=lambda: predict(septal_length,
                                      septal_width,
                                      petal_length,
                                      petal_width,
                                      ann)).grid(row=4, column=1, sticky=tk.W, pady=4)
    master.mainloop()
    tk.mainloop()


species = ['Setosa', 'Versicolor', 'Virginica']

ann = tf.keras.models.load_model("Model/network.h5")
show_gui()

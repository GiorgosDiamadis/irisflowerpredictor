from flask import Flask, render_template, request
import joblib
import os
from Model.load_model import Loader

global model, graph

load_model = Loader()
species = ['Setosa', 'Versicolor', 'Virginica']

os.chdir("./Model")
model, graph = load_model.load()
os.chdir("../")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    s_length = float(request.form["s_length"])
    s_width = float(request.form["s_width"])

    p_length = float(request.form["p_length"])
    p_width = float(request.form["p_width"])

    pred = [[s_length, s_width, p_length, p_width]]
    scaler = joblib.load("Model/scaler.sc")
    prediction = model.predict(scaler.transform(pred))

    maximum = -1
    for i in range(3):
        if prediction[0][i] > prediction[0][maximum]:
            maximum = i

    return render_template("prediction.html", name="species[maximum]")


if __name__ == '__main__':
    app.run()

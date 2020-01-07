# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:18:24 2020

@author: rgottimu
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ['POST'])
def predict():    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text="Empolyee Salary should be $ {}".format(output))


if __name__ == "__main__":
    #app.run(host="localhost", port=int("5000"))
    app.run(debug=True)

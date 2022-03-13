# -*- coding: utf-8 -*-
import flask
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

app = Flask(__name__)

model =pickle.load(open("Random.pkl",'rb'))


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


sc= StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    Gender_Male = 0
    if request.method == 'POST':
        Age=int(request.form['Age'])
        Salary=float(request.form['Salary'])
        Gender_male = request.form['Gender']
        if(Gender_Male == "Male"):
                Male = 0
                Female= 1
        else:
            Female=0
            Male=1
         
    
                     
                     
        prediction = model.predict_proba([[Age,Salary,Gender_Male]])
        
        output='{0:.{1}f}'.format(prediction[0][1],2)
        
        if output >=str(0.5):
            return render_template('index.html',prediction_texts="Hey Congo!!!You Purchaed the product.\nProbability of Purchasing is {}".format(output))
        else:
            return render_template('index.html',prediction_texts="Oh No### You are not eligible for buying the product.\nProbability of purchaing the item is {}".format(output))
              
    else :
        return render_template("index.html")
    
    
    
if __name__  == '__main__':
    app.run(debug=True)

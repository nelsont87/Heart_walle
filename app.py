import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, session
from keras.models import load_model
import secrets
import sys
import tensorflow as tf
import keras
from keras import models, layers
import pickle
from sklearn.metrics import classification_report
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler as ss
from sklearn import preprocessing

app = Flask(__name__)

app.config["SECRET_KEY"] = "first"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/index")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about_us.html")

@app.route("/deep", methods=["GET", "POST"])
def deep_learning():
    if request.method == "POST":
        age = int(request.form["age"])
        gender = int(request.form["gender"])
        height = int(request.form["height"])
        weight = int(request.form["weight"])
        cholesterol = int(request.form["chol"])
        bp_lo = int(request.form["bp_lo"])
        bp_hi = int(request.form["bp_hi"])
        glucose = int(request.form["glucose"])
        smoke = int(request.form["smoke"])
        alc = int(request.form["alc"])
        active = int(request.form["active"])

        input_list = []
        input_list.append(age)
        input_list.append(gender)
        input_list.append(height)
        input_list.append(weight)
        input_list.append(bp_hi)
        input_list.append(bp_lo)
        if cholesterol > 240:
            input_list.append(3)
        elif cholesterol > 200:
            input_list.append(2)
        else:
            input_list.append(1)
        if glucose > 126:
            input_list.append(3)
        elif glucose > 100:
            input_list.append(2)
        else:
            input_list.append(1)
        input_list.append(smoke)
        input_list.append(alc)
        input_list.append(active)


        value = []
        cate = []
        if age > 11:
            if cholesterol > 200:
                value.append((cholesterol - 200)* 0.50976)
                cate.append("cholesterol")
        if age < 11:
            if cholesterol > 170:
                value.append((cholesterol - 170)* 0.50976)
                cate.append("cholesterol")
        if gender == 1 :
            if height > 195 and weight > 92.5:
                value.append((weight - 92.5)* 0.0185)
                cate.append("weight")
            elif height > 193 and weight > 89.8:
                value.append((weight - 89.8)* 0.0185)
                cate.append("weight")
            elif height > 191 and weight > 87.5:
                value.append((weight - 87.5)* 0.0185)
                cate.append("weight")
            elif height > 188 and weight > 84.8:
                value.append((weight - 84.8)* 0.0185)
                cate.append("weight")
            elif height > 185 and weight > 82.5:
                value.append((weight - 82.5)* 0.0185)
                cate.append("weight")
            elif height > 183 and weight > 79.8:
                value.append((weight - 79.8)* 0.0185)
                cate.append("weight")
            elif height > 180 and weight > 77.5:
                value.append((weight - 77.5)* 0.0185)
                cate.append("weight")
            elif height > 178 and weight > 74.8:
                value.append((weight - 74.8)* 0.0185)
                cate.append("weight")
            elif height > 175 and weight > 72.6:
                value.append((weight - 72.6)* 0.0185)
                cate.append("weight")
            elif height > 173 and weight > 69.8:
                value.append((weight - 69.8)* 0.0185)
                cate.append("weight")
            elif height > 170 and weight > 67.6:
                value.append((weight - 67.6)* 0.0185)
                cate.append("weight")
            elif height > 168 and weight > 64.8:
                value.append((weight - 64.8)* 0.0185)
                cate.append("weight")
            elif height > 165 and weight > 62.6:
                value.append((weight - 62.6)* 0.0185)
                cate.append("weight")
            elif height > 163 and weight > 59.9:
                value.append((weight - 59.9)* 0.0185)
                cate.append("weight")
            elif height > 160 and weight > 57.6:
                value.append((weight - 57.6)* 0.0185)
                cate.append("weight")
            elif height > 157 and weight > 54.9:
                value.append((weight - 54.9)* 0.0185)
                cate.append("weight")
            elif height > 155 and weight > 52.6:
                value.append((weight - 52.6)* 0.0185)
                cate.append("weight")
            elif height > 152 and weight > 49.9:
                value.append((weight - 49.9)* 0.0185)
                cate.append("weight")
            elif height > 150 and weight > 47.6:
                value.append((weight - 47.6)* 0.0185)
                cate.append("weight")
            elif height > 147 and weight > 44.9:
                value.append((weight - 44.9)* 0.0185)
                cate.append("weight")
            elif height > 145 and weight > 42.6:
                value.append((weight - 42.6)* 0.0185)
                cate.append("weight")
            elif height > 142 and weight > 39.9:
                value.append((weight - 39.9)* 0.0185)
                cate.append("weight")
            elif height > 140 and weight > 37.6:
                value.append((weight - 37.6)* 0.0185)
                cate.append("weight")
            elif height <= 137 and weight > 34.9:
                value.append((weight - 34.9)* 0.0185)
                cate.append("weight")
        else:
            if height > 195 and weight > 103.8:
                value.append((weight - 103.8)* 0.0185)
                cate.append("weight")
            elif height > 193 and weight > 100.6:
                value.append((weight - 100.6)* 0.0185)
                cate.append("weight")
            elif height > 191 and weight > 98:
                value.append((weight - 98)* 0.0185)
                cate.append("weight")
            elif height > 188 and weight > 94.8:
                value.append((weight - 94.8)* 0.0185)
                cate.append("weight")
            elif height > 185 and weight > 91.6:
                value.append((weight - 91.6)* 0.0185)
                cate.append("weight")
            elif height > 183 and weight > 88.9:
                value.append((weight - 88.9)* 0.0185)
                cate.append("weight")
            elif height > 180 and weight > 85.7:
                value.append((weight - 85.7)* 0.0185)
                cate.append("weight")
            elif height > 178 and weight > 83:
                value.append((weight - 83)* 0.0185)
                cate.append("weight")
            elif height > 175 and weight > 79.8:
                value.append((weight - 79.8)* 0.0185)
                cate.append("weight")
            elif height > 173 and weight > 76.6:
                value.append((weight - 76.6)* 0.0185)
                cate.append("weight")
            elif height > 170 and weight > 73.9:
                value.append((weight - 73.9)* 0.0185)
                cate.append("weight")
            elif height > 168 and weight > 70.7:
                value.append((weight - 70.7)* 0.0185)
                cate.append("weight")
            elif height > 165 and weight > 68:
                value.append((weight - 68)* 0.0185)
                cate.append("weight")
            elif height > 163 and weight > 64.8:
                value.append((weight - 64.8)* 0.0185)
                cate.append("weight")
            elif height > 160 and weight > 61.6:
                value.append((weight - 61.6)* 0.0185)
                cate.append("weight")
            elif height > 157 and weight > 58.9:
                value.append((weight - 58.9)* 0.0185)
                cate.append("weight")
            elif height > 155 and weight > 55.8:
                value.append((weight - 55.8)* 0.0185)
                cate.append("weight")
            elif height > 152 and weight > 53:
                value.append((weight - 53)* 0.0185)
                cate.append("weight")
            elif height > 150 and weight > 49.9:
                value.append((weight - 49.9)* 0.0185)
                cate.append("weight")
            elif height > 147 and weight > 46.7:
                value.append((weight - 46.7)* 0.0185)
                cate.append("weight")
            elif height > 145 and weight > 43.9:
                value.append((weight - 43.9)* 0.0185)
                cate.append("weight")
            elif height > 142 and weight > 40.8:
                value.append((weight - 40.8)* 0.0185)
                cate.append("weight")
            elif height > 140 and weight > 38.1:
                value.append((weight - 38.1)* 0.0185)
                cate.append("weight")
            elif height <= 137 and weight > 34.9:
                value.append((weight - 34.9)* 0.0185)
                cate.append("weight")
        if bp_hi > 120:
            value.append((bp_hi - 120)* 0.0328)
            cate.append("BP")
        if bp_lo > 80:
            value.append((bp_lo - 80)* 0.00036)
            cate.append("BP")
        if glucose > 100:
            value.append((glucose-100)* 0.1512)
            cate.append("glucose")
        if smoke == 1:
            value.append(smoke * 0.0767)
            cate.append("smoking")
        if alc == 1:
            value.append(alc * 0.1069)
            cate.append("alcohol")
        if active == 0:
            value.append(active * 0.3044)
            cate.append("active")

        session["input_list"] = input_list
        session["value"] = value
        session["cate"] = cate
        
    
    return render_template("deep_learning.html")

@app.route("/deepstat")
def deep_stat():

    value_list = session.get("value")
    cate_list = session.get("cate")
    user_in = session.get("input_list")
    Y = 0
    improve = 0
    if value_list is not None:
        for z in range(len(value_list)):
            if value_list[z] > Y:
                Y = value_list[z]
                improve = cate_list[z]
    user_in = np.array([user_in])
    neural_network = load_model("neural_network3.h5", compile=False)
    heart = neural_network.predict(user_in)
    predict = heart.tolist()
    prediction_dict = {
        "improvement": improve,
        "prediction": predict
    }
    keras.backend.clear_session()
    return prediction_dict



@app.route("/svm", methods=["GET", "POST"])
def random_forest():
    if request.method == "POST":
        age = int(request.form["age"])
        gender = int(request.form["gender"])
        height = int(request.form["height"])
        weight = int(request.form["weight"])
        cholesterol = int(request.form["chol"])
        bp_lo = int(request.form["bp_lo"])
        bp_hi = int(request.form["bp_hi"])
        glucose = int(request.form["glucose"])
        smoke = int(request.form["smoke"])
        alc = int(request.form["alc"])
        active = int(request.form["active"])

        input_list = []
        input_list.append(age)
        input_list.append(gender)
        input_list.append(height)
        input_list.append(weight)
        input_list.append(bp_hi)
        input_list.append(bp_lo)
        if cholesterol > 240:
            input_list.append(3)
        elif cholesterol > 200:
            input_list.append(2)
        else:
            input_list.append(1)
        if glucose > 126:
            input_list.append(3)
        elif glucose > 100:
            input_list.append(2)
        else:
            input_list.append(1)
        input_list.append(smoke)
        input_list.append(alc)
        input_list.append(active)

        session["input_list"] = input_list


        value = []
        cate = []
        if age > 11:
            if cholesterol > 200:
                value.append((cholesterol - 200)* 0.381)
                cate.append("cholesterol")
        if age < 11:
            if cholesterol > 170:
                value.append((cholesterol - 170)* 0.381)
                cate.append("cholesterol")
        if gender == 1 :
            if height > 195 and weight > 92.5:
                value.append((weight - 92.5)* 0.1866)
                cate.append("weight")
            elif height > 193 and weight > 89.8:
                value.append((weight - 89.8)* 0.1866)
                cate.append("weight")
            elif height > 191 and weight > 87.5:
                value.append((weight - 87.5)* 0.1866)
                cate.append("weight")
            elif height > 188 and weight > 84.8:
                value.append((weight - 84.8)* 0.1866)
                cate.append("weight")
            elif height > 185 and weight > 82.5:
                value.append((weight - 82.5)* 0.1866)
                cate.append("weight")
            elif height > 183 and weight > 79.8:
                value.append((weight - 79.8)* 0.1866)
                cate.append("weight")
            elif height > 180 and weight > 77.5:
                value.append((weight - 77.5)* 0.1866)
                cate.append("weight")
            elif height > 178 and weight > 74.8:
                value.append((weight - 74.8)* 0.1866)
                cate.append("weight")
            elif height > 175 and weight > 72.6:
                value.append((weight - 72.6)* 0.1866)
                cate.append("weight")
            elif height > 173 and weight > 69.8:
                value.append((weight - 69.8)* 0.1866)
                cate.append("weight")
            elif height > 170 and weight > 67.6:
                value.append((weight - 67.6)* 0.1866)
                cate.append("weight")
            elif height > 168 and weight > 64.8:
                value.append((weight - 64.8)* 0.1866)
                cate.append("weight")
            elif height > 165 and weight > 62.6:
                value.append((weight - 62.6)* 0.1866)
                cate.append("weight")
            elif height > 163 and weight > 59.9:
                value.append((weight - 59.9)* 0.1866)
                cate.append("weight")
            elif height > 160 and weight > 57.6:
                value.append((weight - 57.6)* 0.1866)
                cate.append("weight")
            elif height > 157 and weight > 54.9:
                value.append((weight - 54.9)* 0.1866)
                cate.append("weight")
            elif height > 155 and weight > 52.6:
                value.append((weight - 52.6)* 0.1866)
                cate.append("weight")
            elif height > 152 and weight > 49.9:
                value.append((weight - 49.9)* 0.1866)
                cate.append("weight")
            elif height > 150 and weight > 47.6:
                value.append((weight - 47.6)* 0.1866)
                cate.append("weight")
            elif height > 147 and weight > 44.9:
                value.append((weight - 44.9)* 0.1866)
                cate.append("weight")
            elif height > 145 and weight > 42.6:
                value.append((weight - 42.6)* 0.1866)
                cate.append("weight")
            elif height > 142 and weight > 39.9:
                value.append((weight - 39.9)* 0.1866)
                cate.append("weight")
            elif height > 140 and weight > 37.6:
                value.append((weight - 37.6)* 0.1866)
                cate.append("weight")
            elif height <= 137 and weight > 34.9:
                value.append((weight - 34.9)* 0.1866)
                cate.append("weight")
        else:
            if height > 195 and weight > 103.8:
                value.append((weight - 103.8)* 0.1866)
                cate.append("weight")
            elif height > 193 and weight > 100.6:
                value.append((weight - 100.6)* 0.1866)
                cate.append("weight")
            elif height > 191 and weight > 98:
                value.append((weight - 98)* 0.1866)
                cate.append("weight")
            elif height > 188 and weight > 94.8:
                value.append((weight - 94.8)* 0.1866)
                cate.append("weight")
            elif height > 185 and weight > 91.6:
                value.append((weight - 91.6)* 0.1866)
                cate.append("weight")
            elif height > 183 and weight > 88.9:
                value.append((weight - 88.9)* 0.1866)
                cate.append("weight")
            elif height > 180 and weight > 85.7:
                value.append((weight - 85.7)* 0.1866)
                cate.append("weight")
            elif height > 178 and weight > 83:
                value.append((weight - 83)* 0.1866)
                cate.append("weight")
            elif height > 175 and weight > 79.8:
                value.append((weight - 79.8)* 0.1866)
                cate.append("weight")
            elif height > 173 and weight > 76.6:
                value.append((weight - 76.6)* 0.1866)
                cate.append("weight")
            elif height > 170 and weight > 73.9:
                value.append((weight - 73.9)* 0.1866)
                cate.append("weight")
            elif height > 168 and weight > 70.7:
                value.append((weight - 70.7)* 0.1866)
                cate.append("weight")
            elif height > 165 and weight > 68:
                value.append((weight - 68)* 0.1866)
                cate.append("weight")
            elif height > 163 and weight > 64.8:
                value.append((weight - 64.8)* 0.1866)
                cate.append("weight")
            elif height > 160 and weight > 61.6:
                value.append((weight - 61.6)* 0.1866)
                cate.append("weight")
            elif height > 157 and weight > 58.9:
                value.append((weight - 58.9)* 0.1866)
                cate.append("weight")
            elif height > 155 and weight > 55.8:
                value.append((weight - 55.8)* 0.1866)
                cate.append("weight")
            elif height > 152 and weight > 53:
                value.append((weight - 53)* 0.1866)
                cate.append("weight")
            elif height > 150 and weight > 49.9:
                value.append((weight - 49.9)* 0.1866)
                cate.append("weight")
            elif height > 147 and weight > 46.7:
                value.append((weight - 46.7)* 0.1866)
                cate.append("weight")
            elif height > 145 and weight > 43.9:
                value.append((weight - 43.9)* 0.1866)
                cate.append("weight")
            elif height > 142 and weight > 40.8:
                value.append((weight - 40.8)* 0.1866)
                cate.append("weight")
            elif height > 140 and weight > 38.1:
                value.append((weight - 38.1)* 0.1866)
                cate.append("weight")
            elif height <= 137 and weight > 34.9:
                value.append((weight - 34.9)* 0.1866)
                cate.append("weight")
        if bp_hi > 120:
            value.append((bp_hi - 120)* 7.6517)
            cate.append("BP")
        if bp_lo > 80:
            value.append((bp_lo - 80)* 0.0393)
            cate.append("BP")
        if glucose > 100:
            value.append((glucose-100)* 0.041314)
            cate.append("glucose")
        if smoke == 1:
            value.append(smoke * 0.030607)
            cate.append("smoking")
        if alc == 1:
            value.append(alc * 0.022789)
            cate.append("alcohol")
        if active == 0:
            value.append(active * 0.079956)
            cate.append("active")
        
    return render_template("svm_gs.html")

@app.route("/svmstat")
def svm_stat():

    value_list = session.get("value")
    cate_list = session.get("cate")
    user_in = session.get("input_list")
    Y = 0
    improve = 0
    if value_list is not None:
        for z in range(len(value_list)):
            if value_list[z] > Y:
                Y = value_list[z]
                improve = cate_list[z]
    # user_in = np.array(user_in)
    svm = pickle.load(open("svm_ss_gridsearch.pkl", "rb"))
    heart = svm.predict([user_in])
    predict = heart.tolist()
    prediction_dict = {
        "improvement": improve,
        "prediction": predict
    }
    keras.backend.clear_session()
    return prediction_dict




@app.route("/logreg", methods=["GET", "POST"])
def log_regression():
    if request.method == "POST":
        age = int(request.form["age"])
        gender = int(request.form["gender"])
        height = int(request.form["height"])
        weight = int(request.form["weight"])
        cholesterol = int(request.form["chol"])
        bp_lo = int(request.form["bp_lo"])
        bp_hi = int(request.form["bp_hi"])
        glucose = int(request.form["glucose"])
        smoke = int(request.form["smoke"])
        alc = int(request.form["alc"])
        active = int(request.form["active"])

        # input_list = []
        # input_list.append(age)
        # input_list.append(gender)
        # input_list.append(height)
        # input_list.append(weight)
        # input_list.append(bp_hi)
        # input_list.append(bp_lo)
        # if cholesterol > 240:
        #     input_list.append(3)
        # elif cholesterol > 200:
        #     input_list.append(2)
        # else:
        #     input_list.append(1)
        # if glucose > 126:
        #     input_list.append(3)
        # elif glucose > 100:
        #     input_list.append(2)
        # else:
        #     input_list.append(1)
        # input_list.append(smoke)
        # input_list.append(alc)
        # input_list.append(active)

        # session["input_list"] = input_list

        session["age"] = age
        session["gender"] = gender
        session["height"] = height
        session["weight"] = weight
        session["bp_hi"] = bp_hi
        session["bp_lo"] = bp_lo
        if cholesterol > 240:
            session["cholesterol"] = 3
        elif cholesterol > 200:
            session["cholesterol"] = 2
        else:
            session["cholesterol"] = 1  
        if glucose > 126:
            session["glucose"] = 3
        elif glucose > 100:
            session["glucose"] = 2
        else:
            session["glucose"] = 1
        session["smoke"] = smoke
        session["alc"] = alc
        session["active"] = active

        value = []
        cate = []
        if age > 11:
            if cholesterol > 200:
                value.append((cholesterol - 200)* 0.50976)
                cate.append("cholesterol")
        if age < 11:
            if cholesterol > 170:
                value.append((cholesterol - 170)* 0.50976)
                cate.append("cholesterol")
        if gender == 1 :
            if height > 195 and weight > 92.5:
                value.append((weight - 92.5)* 0.0185)
                cate.append("weight")
            elif height > 193 and weight > 89.8:
                value.append((weight - 89.8)* 0.0185)
                cate.append("weight")
            elif height > 191 and weight > 87.5:
                value.append((weight - 87.5)* 0.0185)
                cate.append("weight")
            elif height > 188 and weight > 84.8:
                value.append((weight - 84.8)* 0.0185)
                cate.append("weight")
            elif height > 185 and weight > 82.5:
                value.append((weight - 82.5)* 0.0185)
                cate.append("weight")
            elif height > 183 and weight > 79.8:
                value.append((weight - 79.8)* 0.0185)
                cate.append("weight")
            elif height > 180 and weight > 77.5:
                value.append((weight - 77.5)* 0.0185)
                cate.append("weight")
            elif height > 178 and weight > 74.8:
                value.append((weight - 74.8)* 0.0185)
                cate.append("weight")
            elif height > 175 and weight > 72.6:
                value.append((weight - 72.6)* 0.0185)
                cate.append("weight")
            elif height > 173 and weight > 69.8:
                value.append((weight - 69.8)* 0.0185)
                cate.append("weight")
            elif height > 170 and weight > 67.6:
                value.append((weight - 67.6)* 0.0185)
                cate.append("weight")
            elif height > 168 and weight > 64.8:
                value.append((weight - 64.8)* 0.0185)
                cate.append("weight")
            elif height > 165 and weight > 62.6:
                value.append((weight - 62.6)* 0.0185)
                cate.append("weight")
            elif height > 163 and weight > 59.9:
                value.append((weight - 59.9)* 0.0185)
                cate.append("weight")
            elif height > 160 and weight > 57.6:
                value.append((weight - 57.6)* 0.0185)
                cate.append("weight")
            elif height > 157 and weight > 54.9:
                value.append((weight - 54.9)* 0.0185)
                cate.append("weight")
            elif height > 155 and weight > 52.6:
                value.append((weight - 52.6)* 0.0185)
                cate.append("weight")
            elif height > 152 and weight > 49.9:
                value.append((weight - 49.9)* 0.0185)
                cate.append("weight")
            elif height > 150 and weight > 47.6:
                value.append((weight - 47.6)* 0.0185)
                cate.append("weight")
            elif height > 147 and weight > 44.9:
                value.append((weight - 44.9)* 0.0185)
                cate.append("weight")
            elif height > 145 and weight > 42.6:
                value.append((weight - 42.6)* 0.0185)
                cate.append("weight")
            elif height > 142 and weight > 39.9:
                value.append((weight - 39.9)* 0.0185)
                cate.append("weight")
            elif height > 140 and weight > 37.6:
                value.append((weight - 37.6)* 0.0185)
                cate.append("weight")
            elif height <= 137 and weight > 34.9:
                value.append((weight - 34.9)* 0.0185)
                cate.append("weight")
        else:
            if height > 195 and weight > 103.8:
                value.append((weight - 103.8)* 0.0185)
                cate.append("weight")
            elif height > 193 and weight > 100.6:
                value.append((weight - 100.6)* 0.0185)
                cate.append("weight")
            elif height > 191 and weight > 98:
                value.append((weight - 98)* 0.0185)
                cate.append("weight")
            elif height > 188 and weight > 94.8:
                value.append((weight - 94.8)* 0.0185)
                cate.append("weight")
            elif height > 185 and weight > 91.6:
                value.append((weight - 91.6)* 0.0185)
                cate.append("weight")
            elif height > 183 and weight > 88.9:
                value.append((weight - 88.9)* 0.0185)
                cate.append("weight")
            elif height > 180 and weight > 85.7:
                value.append((weight - 85.7)* 0.0185)
                cate.append("weight")
            elif height > 178 and weight > 83:
                value.append((weight - 83)* 0.0185)
                cate.append("weight")
            elif height > 175 and weight > 79.8:
                value.append((weight - 79.8)* 0.0185)
                cate.append("weight")
            elif height > 173 and weight > 76.6:
                value.append((weight - 76.6)* 0.0185)
                cate.append("weight")
            elif height > 170 and weight > 73.9:
                value.append((weight - 73.9)* 0.0185)
                cate.append("weight")
            elif height > 168 and weight > 70.7:
                value.append((weight - 70.7)* 0.0185)
                cate.append("weight")
            elif height > 165 and weight > 68:
                value.append((weight - 68)* 0.0185)
                cate.append("weight")
            elif height > 163 and weight > 64.8:
                value.append((weight - 64.8)* 0.0185)
                cate.append("weight")
            elif height > 160 and weight > 61.6:
                value.append((weight - 61.6)* 0.0185)
                cate.append("weight")
            elif height > 157 and weight > 58.9:
                value.append((weight - 58.9)* 0.0185)
                cate.append("weight")
            elif height > 155 and weight > 55.8:
                value.append((weight - 55.8)* 0.0185)
                cate.append("weight")
            elif height > 152 and weight > 53:
                value.append((weight - 53)* 0.0185)
                cate.append("weight")
            elif height > 150 and weight > 49.9:
                value.append((weight - 49.9)* 0.0185)
                cate.append("weight")
            elif height > 147 and weight > 46.7:
                value.append((weight - 46.7)* 0.0185)
                cate.append("weight")
            elif height > 145 and weight > 43.9:
                value.append((weight - 43.9)* 0.0185)
                cate.append("weight")
            elif height > 142 and weight > 40.8:
                value.append((weight - 40.8)* 0.0185)
                cate.append("weight")
            elif height > 140 and weight > 38.1:
                value.append((weight - 38.1)* 0.0185)
                cate.append("weight")
            elif height <= 137 and weight > 34.9:
                value.append((weight - 34.9)* 0.0185)
                cate.append("weight")
        if bp_hi > 120:
            value.append((bp_hi - 120)* 0.0328)
            cate.append("BP")
        if bp_lo > 80:
            value.append((bp_lo - 80)* 0.00036)
            cate.append("BP")
        if glucose > 100:
            value.append((glucose-100)* 0.1512)
            cate.append("glucose")
        if smoke == 1:
            value.append(smoke * 0.0767)
            cate.append("smoking")
        if alc == 1:
            value.append(alc * 0.1069)
            cate.append("alcohol")
        if active == 0:
            value.append(active * 0.3044)
            cate.append("active")

    return render_template("logistic_regression.html")

@app.route("/logstat")
def log_stat():

    value_list = session.get("value")
    cate_list = session.get("cate")
    Y = 0
    improve = 0
    if value_list is not None:
        for z in range(len(value_list)):
            if value_list[z] > Y:
                Y = value_list[z]
                improve = cate_list[z]


    Xage = session.get("age")
    Xgender = session.get("gender")
    Xheight = session.get("height")
    Xweight = session.get("weight")
    Xbph = session.get("bp_hi")
    Xbpl = session.get("bp_lo")
    Xchol = session.get("cholesterol")
    Xgluc = session.get("glucose")
    Xsmoke = session.get("smoke")
    Xalc = session.get("alc")
    Xactive = session.get("active")

    X = []
    X.append(Xage)
    X.append(Xgender)
    X.append(Xheight)
    X.append(Xweight)
    X.append(Xbph)
    X.append(Xbpl)
    X.append(Xchol)
    X.append(Xgluc)
    X.append(Xsmoke)
    X.append(Xalc)
    X.append(Xactive)

    X = np.array(X)
    # user_in = user_in.reshape(1,-1)
    print(X)
    log = pickle.load(open("logistical_regression_grid_search.pkl", "rb"))
    heart = log.predict((X.reshape(1,-1)))
    keras.backend.clear_session()
    predict = heart
    prediction_dict = {
        "improvement": improve,
        "prediction": predict
    }

    return prediction_dict

if __name__ == "__main__":
    app.run(debug=True)
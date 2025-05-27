
# A very simple Flask Hello World app for you to get started with...
from utils.constants import constants
from flask import Flask, render_template, request, flash
import warnings
import subprocess
import sys
import net_benefit as nb
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = {'csv'}
metrics = {}
input_params = {}
decision_curve = {}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

@app.context_processor
def inject_constants():
    return dict(constants=constants)

@app.route('/', methods=["POST", "GET"])
def home():
    return render_template("home.html")



'''
when user try the tool predefinited data
calculates metrics with predefinited data
'''
@app.route("/test_mode/", methods=["POST", "GET"])
def test_mode():
    home_path = sys.path
    # df = pd.read_csv("mysite/static/testfiles/test.csv")
    df = pd.read_csv("static/testfiles/test.csv")
    init_input_params()
    get_input_params(df) #converts each column in the file into numpy arrays
    compute_metrics(input_params["y_true"], input_params["y_proba"], input_params["relevance"], input_params["threshold"])
    compute_decision_curve(input_params["y_true"], input_params["y_proba"])
    return render_template("results.html", metrics=metrics, decision_curve=decision_curve)



'''
when user try the tool with his own data
calculates metrics with data passed in input by the user
'''
@app.route("/custom_mode/", methods=["POST", "GET"])
def custom_mode():
    #get input file, store it in a tmp folder and read his content
    file = request.files['file1']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], "input.csv"))
    df = pd.read_csv("/tmp/input.csv")
    init_input_params()
    get_input_params(df)  #separate input from data frame and put in dictonary
    #if the threshold is passed as scalar
    ths_scalar = request.form['ths_scalar']
    if ths_scalar != '':
        input_params["threshold"] = float(ths_scalar)
    compute_metrics(input_params["y_true"], input_params["y_proba"], input_params["relevance"], input_params["threshold"])
    compute_decision_curve(input_params["y_true"], input_params["y_proba"])
    return render_template("results.html", metrics=metrics, decision_curve=decision_curve)


'''
compute all available metrics and store them in a dictionary
'''
def compute_metrics(y_true, y_proba, relevances, ths):
    y_pred = (y_proba >= ths).astype(int) #convert probabilities into binary prediction
    metrics["wu"] = nb.wu(y_true, y_proba, ths, relevances)
    metrics["nb25"] = nb.nb(y_true, y_proba,0.25)
    metrics["nb50"] = nb.nb(y_true, y_proba,0.50)
    metrics["nb75"] = nb.nb(y_true, y_proba,0.75)
    metrics["acc"] = accuracy_score(y_true, y_pred)
    metrics["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
    try:
    	metrics["auc"] = roc_auc_score(y_true, y_pred)
    except ValueError: #undefined AUC
    	metrics["auc"] = '-'
    metrics["sens"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["spec"] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred)



def init_input_params():
    input_params["y_true"] = None
    input_params["y_proba"] = None
    input_params["relevance"] = None
    input_params["threshold"] = 0.5


'''
converts each column in the file (param) into numpy arrays
'''
def get_input_params(df):
    header = df.columns.values
    for param in header:
        input_params[param] = df[param].to_numpy()

'''
compute net benefit decision curve for different threshold
'''
def compute_decision_curve(y_true, y_proba):
    axis = {"x":[], "y":[]}
    thresholds = [round(th,1) for th in np.arange(0, 1, 0.1)]
    net_benefits = [nb.nb(y_true, y_proba,th) for th in thresholds]
    treat_all = [nb.nb(y_true, np.ones_like(y_proba),th) for th in thresholds]
    decision_curve["Standardized_net_benefit"] = {"x":thresholds, "y":net_benefits}
    decision_curve["Treat_none"] = {"x":thresholds, "y":[0]*len(net_benefits)}
    decision_curve["Treat_all"] = {"x":thresholds, "y":treat_all}


@app.route('/input')
def input():
    return render_template('input.html')


if __name__ == "__main__":
    app.run(debug=True, port=5002)
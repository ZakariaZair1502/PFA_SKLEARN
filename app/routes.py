from pyexpat import model
from flask import Blueprint, render_template,request
import numpy as np
import os
import app
from app.regression_models import model_train, model_evaluat
import joblib as jb
bp = Blueprint('main', __name__)
global x,y,x_test,algo,md
@bp.route('/', methods=['GET', 'POST'])
def index():
    algo = None
    coef_ = None
    y_pred = None
    if request.method == 'POST':
        algo = request.form.get('algo')
        x = request.form.get('x')
        x = x.strip('[]')
        x = [list(map(float, item.split(','))) for item in x.split('],[')]
        x = np.array(x)
        y = request.form.get('y')
        y = y.strip('[]')
        y = list(map(float, y.split(',')))
        y = np.array(y)
        x_test = request.form.get('x_test')
        x_test = x_test.strip('[]')
        x_test = [list(map(float, item.split(','))) for item in x_test.split('],[')]
        x_test = np.array(x_test)
        if algo:
            try:
                md = model_train(x,y,algo)
                y_pred = md.predict(x_test) 
                coef_ = md.coef_
                coef_ = str(coef_)
                y_pred = str(y_pred)

            except ValueError as e:
                coef_ = str(e)
        coef_ = str(coef_)
        y_pred = str(y_pred)
    
    return render_template('index.html', algo=algo, coef_=coef_, y_pred = y_pred)

@bp.route('/evaluate',methods=['GET', 'POST'])
def evaluate():
    algo = request.form.get('algo')
    x = np.array([list(map(float, item.split(','))) for item in request.form.get('x').strip('[]').split('],[')])
    y = np.array(list(map(float, request.form.get('y').strip('[]').split(','))))
    x_test = np.array([list(map(float, item.split(','))) for item in request.form.get('x_test').strip('[]').split('],[')])

    score, mse, mae = model_evaluat(x, y, x_test, algo)

    return render_template('index.html', score=score, mse=mse, mae=mae)
    
@bp.route('/save',methods=['GET', 'POST'])
def save():
    algo = request.form.get('algo')
    x = np.array([list(map(float, item.split(','))) for item in request.form.get('x').strip('[]').split('],[')])
    y = np.array(list(map(float, request.form.get('y').strip('[]').split(','))))

    model_trained = model_train(x, y, algo)
    save_path = "app/static/uploads/"
    print(bp.root_path)
    os.makedirs(save_path, exist_ok=True) 
    if model_trained:
        model_file = os.path.join(save_path, "model.pkl")
        jb.dump(model_trained,model_file)
        save = "Modele sauvegardé"
    else : 
        save = "Modele non sauvegardé"
    return render_template('index.html',save = save)


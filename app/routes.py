from flask import Blueprint, render_template,request
import numpy as np
from app.regression_models import model_train 
bp = Blueprint('main', __name__)

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

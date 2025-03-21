from fileinput import filename
from pyexpat import model
from flask import Blueprint, render_template,request
import numpy as np
import pandas as pd
import os
from app.regression_models import model_train, model_evaluat
import joblib as jb

bp = Blueprint('main', __name__)

params_cache = {}
filename = ""
UPLOAD_FOLDER = "app/static/datasets"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@bp.route('/', methods=['GET', 'POST'])
def index():
    global params_cache 
    global filename
    algo = None
    params = None
    if request.method == 'POST':
        file = request.files.get('dataset') 
        if file and file.filename.endswith('.csv'):
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath) 
        algo = request.form.get('algo')
        cache_key = f"{file.filename}_{algo}"
        if cache_key in params_cache:
            params = params_cache[cache_key]
        else:
            try:
                params = model_train(filepath, algo)
                params_cache[cache_key] = params  # Stocker en cache
            except ValueError as e:
                return render_template('index.html', params=str(e))    
    return render_template('index.html', algo=algo, params = params)

@bp.route('/evaluate',methods=['GET', 'POST'])
def evaluate():
    algo = request.form.get('algo')
    if filename and algo:
        cache_key = f"{filename}_{algo}"
        if cache_key in params_cache:
            params = params_cache[cache_key]
        else:
            filepath = os.path.join(UPLOAD_FOLDER,filename)
            params = model_train(filepath, algo)
            params_cache[cache_key] = params  # Stocker en cache

        metrics = model_evaluat(params)
        return render_template('index.html',metrics=metrics, filename=filename, algo=algo)
    return render_template('index.html')
    
@bp.route('/save',methods=['GET', 'POST'])
def save():
    algo = request.form.get('algo')

    if filename and algo:
        cache_key = f"{filename}_{algo}"
        if cache_key in params_cache:
            params = params_cache[cache_key]
        else:
            filepath = os.path.join(UPLOAD_FOLDER,filename)
            params = model_train(filepath, algo)
            params_cache[cache_key] = params  # Stocker en cache

        save_path = "app/static/uploads/"
        os.makedirs(save_path, exist_ok=True)
        model_file = os.path.join(save_path, "model.pkl")
        jb.dump(params['md'], model_file)
        return render_template('index.html', save="Modèle sauvegardé avec succès!", filename=filename, algo=algo)

    return render_template('index.html', save="Erreur lors de la sauvegarde du modèle.")


from flask import Blueprint, render_template, request
import os
from app.regression_models import model_train, model_evaluat
import joblib as jb
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend non interactif
import matplotlib.pyplot as plt
from flask import send_file
import numpy as np
bp = Blueprint('main', __name__)

params_cache = {}
UPLOAD_FOLDER = "app/static/datasets"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@bp.route('/', methods=['GET', 'POST'])
def index():
    algo = None
    params = None
    predictions = None
    filename = ""

    if request.method == 'POST':
        file = request.files.get('dataset') 
        if file and file.filename.endswith('.csv'):
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            algo = request.form.get('algo')
            if algo:
                cache_key = f"{filename}_{algo}"
                
                if cache_key in params_cache:
                    params = params_cache[cache_key]
                else:
                    try:
                        params = model_train(filepath, algo)
                        params_cache[cache_key] = params  # Stocker en cache
                    except ValueError as e:
                        return render_template('index.html', error=str(e))    

                if params:
                    predictions = [round(value, 1) for value in params['md'].predict(params['x_test']).tolist()]

    return render_template('index.html', algo=algo, params=params, filename=filename, predictions=predictions)

@bp.route('/evaluate',methods=['GET', 'POST'])
def evaluate():
    algo = request.form.get('algo')
    filename = request.form.get('filename')
    if filename and algo:
        cache_key = f"{filename}_{algo}"
        if cache_key in params_cache:
            params = params_cache[cache_key]
        else:
            filepath = os.path.join(UPLOAD_FOLDER,filename)
            params = model_train(filepath, algo)
            params_cache[cache_key] = params  # Stocker en cache

        params = model_evaluat(params)
        metrics = params['metrics']
        return render_template('index.html',metrics=metrics, filename=filename, algo=algo)
    return render_template('index.html')
    
@bp.route('/save', methods=['POST'])
def save():
    filename = request.form.get('filename')
    algo = request.form.get('algo')

    if not filename or not algo:
        return render_template('index.html', save="Erreur : fichier ou algorithme non spécifié.")

    cache_key = f"{filename}_{algo}"

    if cache_key in params_cache:
        params = params_cache[cache_key]
    else:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            return render_template('index.html', save="Erreur : fichier introuvable.")
        
        try:
            params = model_train(filepath, algo)
            params_cache[cache_key] = params  # Stocker en cache
        except ValueError as e:
            return render_template('index.html', save=f"Erreur : {str(e)}")

    # Sauvegarde du modèle
    save_path = "app/static/uploads/"
    os.makedirs(save_path, exist_ok=True)
    model_file = os.path.join(save_path, f"{filename}_{algo}.pkl")
    jb.dump(params['md'], model_file)

    return render_template('index.html', save="Modèle sauvegardé avec succès!", filename=filename, algo=algo)

@bp.route('/plot_error_curve', methods=['GET', 'POST'])
def plot_error_curve():
    # Exemple de données (remplacer par des données réelles)
    algo = request.form.get('algo')
    filename = request.form.get('filename')
    if filename and algo:
        cache_key = f"{filename}_{algo}"
        if cache_key in params_cache:
            params = params_cache[cache_key]
        else:
            filepath = os.path.join(UPLOAD_FOLDER,filename)
            params = model_train(filepath, algo)
            params_cache[cache_key] = params  # Stocker en cache

        params = model_evaluat(params)

    # Création de la figure
    plt.figure(figsize=(8, 6))
    plt.plot(params['y_test'], label='y_test', color='blue', marker='o')
    plt.plot(params['md'].predict(params['x_test']), label='y_pred', color='red', marker='x')
    plt.title('Courbe d\'erreur (MSE) et (MAE) par itération')
    plt.xlabel('Itération')
    plt.ylabel('Erreur MSE')
    plt.legend()
    plt.grid(True)
    save_path = "error_plots/"
    os.makedirs(save_path, exist_ok=True)
    image_path = os.path.join(save_path, f"{filename}_{algo}.png")
    plt.savefig(image_path)  # Sauvegarder l'image
    plt.close()
    img_filename = f"{filename}_{algo}.png"
    img_path = f"error_plots/{img_filename}"
    # Envoi de l'image en tant que réponse
    return render_template('index.html', img_path = image_path, filename=filename, algo=algo)

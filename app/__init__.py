from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')  # Charger la config

    from app.routes import bp  # Importer le blueprint des routes
    app.register_blueprint(bp)  # Enregistrer les routes

    return app

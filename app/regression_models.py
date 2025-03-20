from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor

ALGORITHMS = {
    'Linear Regression': LinearRegression,
    'SVR': SVR,
    'Decision Tree': DecisionTreeRegressor,
    'Ridge': Ridge,
    'Lasso': Lasso,
    'Elastic Net': ElasticNet,
    'Random Forest': RandomForestRegressor,
    'Gradient Boosting': GradientBoostingRegressor
}
def model_train(x,y,model_name):
    if model_name not in ALGORITHMS:
        raise ValueError(f"Modèle {model_name} non valide")
    model_class = ALGORITHMS[model_name]
    md = model_class()
    md.fit(x,y)
    return md


from numpy import mean
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
def model_evaluat(x,y,x_test,model_name):
    md = model_train(x,y,model_name)
    score = md.score(x,y)
    mse = mean_squared_error(y,md.predict(x_test))
    mae = mean_absolute_error(y,md.predict(x_test))
    return score,mse,mae


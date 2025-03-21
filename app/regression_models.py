import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.model_selection import train_test_split
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
def model_train(file,model_name):
    if model_name not in ALGORITHMS:
        raise ValueError(f"Modèle {model_name} non valide")
    model_class = ALGORITHMS[model_name]
    md = model_class()
    data = pd.read_csv(file)
    x = data.iloc[:,1:-1].values 
    y = data.iloc[:, -1].values
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    md.fit(x_train,y_train)
    params = {
        'x_train':x_train,
        'y_train':y_train,
        'x_test': x_test,
        'y_test': y_test,
        'md': md,
        'algo': model_name
    }
    return params

def model_evaluat(params):
    params['metrics'] = {
        'score': params['md'].score(params['x_train'], params['y_train']),
        'mse': mean_squared_error(params['y_test'], params['md'].predict(params['x_test'])),
        'mae': mean_absolute_error(params['y_test'], params['md'].predict(params['x_test']))
    }
    return params




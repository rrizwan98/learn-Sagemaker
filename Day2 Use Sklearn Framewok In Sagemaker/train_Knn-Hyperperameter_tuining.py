from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import pandas as pd
import os, argparse
import numpy as np

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, 'knn_model.joblib'))
    return model

def load_dataset(path):
    # Load dataset
    data = pd.read_csv(path)
    # Split samples and labels
    x = data.drop('LUNG_CANCER', axis=1)
    y = data.LUNG_CANCER
    return x,y

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--max_depth",
        type=int,
    )
    
    parser.add_argument("--random_state", type=int)
    parser.add_argument('--n_estimators', type=int, default=250)
                        
    
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()
    
    max_depth= args.max_depth
    random_state= args.random_state
    n_estimators =args.n_estimators
 
    model_dir = args.model_dir
    training_dir = args.training_dir
    validation_dir = args.validation
    
    print(training_dir)
    x_train, y_train = load_dataset(os.path.join(training_dir, 'sagemaker_training_dataset.csv'))
    x_val, y_val     = load_dataset(os.path.join(validation_dir, 'sagemaker_validation_dataset.csv')) 
    
    from sklearn.model_selection import GridSearchCV

    param_grid = {'n_neighbors' : [2,4],
                  'weights' : ['uniform','distance'],
                  'metric' : ['minkowski','euclidean']}
    
    cls = KNeighborsClassifier()  
    g_search = GridSearchCV(estimator = cls, param_grid = param_grid,cv = 2, n_jobs = 1, verbose = 0, return_train_score=True)
    
    g_search.fit(x_train, y_train)
    
    auc = g_search.score(x_val, y_val)
    y_pred=g_search.predict(x_val)
    
    auc = g_search.score(x_val, y_val)
    
    print('***************confusion_matrix***************\n')
    print(confusion_matrix(y_val,y_pred))
    print('\n***************accuracy_score***************\n')
    print(accuracy_score(y_val,y_pred))
    print('\n***************classification_report***************\n')
    print(classification_report(y_val,y_pred))
    print('\n***************AUC Score***************\n')
    print("AUC ", auc)
    print('\n***************best_parameters***************\n')
    print('Best Parameters: ',g_search.best_params_)
    print('\n*********************************************\n')
    
    model = os.path.join(model_dir, 'knn_model.joblib')
    joblib.dump(g_search, model)
    # See https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
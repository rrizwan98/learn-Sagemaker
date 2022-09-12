import os, argparse
import xgboost as xgb
import pandas as pd

def model_fn(model_dir):
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, 'xgb.model'))
    return model

def load_dataset(path):
    # Load dataset
    data = pd.read_csv(path)
    # Split samples and labels
    x = data.drop('LUNG_CANCER', axis=1)
    y = data.LUNG_CANCER
    return x,y

if __name__ == '__main__':
    
    print("XGBoost", xgb.__version__)

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--max_depth",
        type=int,
    )
    
    parser.add_argument("--eta", type=float)
    parser.add_argument("--gamma", type=int)
    parser.add_argument("--min_child_weight", type=int)
    parser.add_argument("--subsample", type=float)
    parser.add_argument("--verbosity", type=int)
    parser.add_argument("--objective", type=str)
    parser.add_argument("--num_round", type=int)
    parser.add_argument("--tree_method", type=str, default="auto")
    parser.add_argument("--predictor", type=str, default="auto")
    parser.add_argument('--n_estimators', type=int, default=250)
                        
    
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()
    # model hyperperameters
    
    max_depth= args.max_depth
    eta= args.eta
    gamma= args.gamma
    min_child_weight= args.min_child_weight
    subsample= args.subsample
    verbosity= args.verbosity
    objective= args.objective
    tree_method= args.tree_method
    predictor= args.predictor
    n_estimators =args.n_estimators
    # Dataset path
    
    model_dir = args.model_dir
    training_dir = args.training_dir
    validation_dir = args.validation
    
    print(training_dir)
    x_train, y_train = load_dataset(os.path.join(training_dir, 'sagemaker_training_dataset.csv'))
    x_val, y_val     = load_dataset(os.path.join(validation_dir, 'sagemaker_validation_dataset.csv'))
    
#     x_train, y_train = load_dataset(training_dir)
#     x_val, y_val     = load_dataset(validation_dir)
    
    cls = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=n_estimators,
        max_depth=max_depth,
        eta=eta,gamma=gamma, min_child_weight=min_child_weight, subsample=subsample, verbosity=verbosity, tree_method =tree_method,
        predictor = predictor)
    
    cls.fit(x_train, y_train)
    auc = cls.score(x_val, y_val)
    print("AUC ", auc)
    
    cls.save_model(os.path.join(model_dir, 'xgb.model'))
    # See https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
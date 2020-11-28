import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import util
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import hypertune

def train_brt(args):
    #load data
    features, labels = util.load_data(args.features_file, args.labels_file)
    #convert time series data to supervised learning problem
    features, labels = util.convert_time_series_to_array(features, labels, args.label_width, args.input_width, args.input_stride, args.sampling_rate)
    features = features.reshape(features.shape[0], features.shape[1] * features.shape[2])
    labels = labels.reshape(labels.shape[0], labels.shape[1] * labels.shape[2])
    #initialize model
    brt = MultiOutputRegressor(HistGradientBoostingRegressor(learning_rate=args.learning_rate, max_depth=args.max_depth, verbose=1))
    #initialize metrics
    metrics_names = ["RMSE"]
    metrics = [tf.metrics.RootMeanSquaredError()]
    #determine if tuning or training final model
    if(args.cross_validation): #tuning using cross validation
        errors = util.cross_validation(features, labels, brt, args.n_splits, metrics) #get the errors from the CV
        err_means = errors.mean(axis=0) #get means
        #output mean error to of each metric for hypertuning
        print(err_means)
        hpt = hypertune.HyperTune() 
        for i in range(len(metrics)):
            hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag=metrics_names[i], metric_value=err_means[i])
    else: #training for prediction
        brt.fit(features, labels) #fit the model using all the data
        joblib.dump(brt, args.model_name) #save the model file
        util.save_model(args.model_dir, args.model_name) #upload the saved model to the cloud
        
def get_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('true'):
            return True
        elif v.lower() in ('false'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument("--model-name",
                            type=str,
                            default="model",
                            help="What to name the saved model file")
    parser.add_argument("--features-file",
                            required=True,
                            help="path to features file")
    parser.add_argument("--labels-file",
                            required=True,
                            help="path to labels file")
    parser.add_argument("--cross-validation",
                            type=str2bool, 
                            default=True,
                            help="whether to do cross validation or not (default: True)")
    parser.add_argument("--n-splits",
                            type=int,
                            default=2,
                            help="number of splits for cross validation (default: 2)")
    parser.add_argument("--label-width",
                            type=int,
                            default=1,
                            help="number of days to predict into the future (default: 1)")
    parser.add_argument("--input-width",
                            type=int,
                            default=10,
                            help="width of input timestep (default: 10)")
    parser.add_argument("--input-stride",
                            type=int,
                            default=1,
                            help="stride of the input (default: 1)")
    parser.add_argument("--sampling-rate",
                            type=int,
                            default=1,
                            help="how often within a timestep the data is sampled (default: 1)")
    parser.add_argument("--n-estimators",
                            type=int,
                            default=100,
                            help="number of boosting stages to perform (default: 100)")
    parser.add_argument("--learning-rate",
                            type=float,
                            default=0.1,
                            help="learning rate shrinks the contribution of each estimator (default: 0.1)")
    parser.add_argument("--max-depth",
                            type=int,
                            default=3,
                            help="max depth of each individual estimator (default: 3)")
    parser.add_argument("--subsample",
                            type=float,
                            default=1.0,
                            help="the fraction of samples to be used for fitting the learners (less than 1.0 results in stochastic gradient boosting) (default: 1.0)")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    train_brt(args)

if __name__ == "__main__":
    main()
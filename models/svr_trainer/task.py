import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import util
from sklearn.svm import LinearSVR 
from sklearn.multioutput import MultiOutputRegressor
import joblib
import hypertune

def train_lr(args):
    #load data
    features, labels = util.load_data(args.features_file, args.labels_file)
    #convert time series data to supervised learning problem
    features, labels = util.convert_time_series_to_array(features, labels, args.label_width, args.input_width, args.input_stride, args.sampling_rate)
    features = features.reshape(features.shape[0], features.shape[1] * features.shape[2])
    labels = labels.reshape(labels.shape[0], labels.shape[1] * labels.shape[2])
    #initialize model
    lr = MultiOutputRegressor(LinearSVR(fit_intercept=0, tol=args.tol, verbose=1), n_jobs=-1)
    #initialize metrics
    metrics_names = ["RMSE"]
    metrics = [tf.metrics.RootMeanSquaredError()]
    #determine if tuning or training final model
    if(args.cross_validation): #tuning using cross validation
        errors = util.cross_validation(features, labels, lr, args.n_splits, metrics) #get the errors from the CV
        err_means = errors.mean(axis=0) #get means
        #output mean error to of each metric for hypertuning
        print(err_means)
        hpt = hypertune.HyperTune() 
        for i in range(len(metrics)):
            hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag=metrics_names[i], metric_value=err_means[i])
    else: #training for prediction
        lr.fit(features, labels) #fit the model using all the data
        joblib.dump(lr, args.model_name) #save the model file
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
    parser.add_argument("--kernel",
                            type=str,
                            choices=["linear", "poly", "rbf", "sigmoid", "precomputed"],
                            default="rbf",
                            help="type of kernel to use(default: rbf)")
    parser.add_argument("--degree",
                            type=int,
                            default=3,
                            help="degree of the polynomial kernel function, ignored by other kernel types (default: 3)")
    parser.add_argument("--gamma",
                            type=str,
                            choices=["scale", "auto"],
                            default="scale",
                            help="kernel coefficient for rbf, poly and sigmoid (default: scale)")
    parser.add_argument("--coef",
                            type=float,
                            default=0.0,
                            help="independent term in kernel function for poly and sigmoid kerels (default: 0.0)")
    parser.add_argument("--tol",
                            type=float,
                            default=1e-3,
                            help="tolerance for stopping (default: 0.001)")
    parser.add_argument("--c",
                            type=float,
                            default=1.0,
                            help="regularization parameter, must be positive (default: 1.0)")
    parser.add_argument("--epsilon",
                            type=float,
                            default=0.1,
                            help="epsilon for the epsilon-svr model (default: 0.1)")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    train_lr(args)

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import util
import joblib
import hypertune

def train_ann(args):
    #load data
    features, labels = util.load_data(args.features_file, args.labels_file)
    #convert time series data to supervised learning problem
    features, labels = util.convert_time_series_to_array(features, labels, args.label_width, args.input_width, args.input_stride, args.sampling_rate)
    features = features.reshape(features.shape[0], features.shape[1] * features.shape[2])
    labels = labels.reshape(labels.shape[0], labels.shape[1] * labels.shape[2])
    #initialize model
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(args.n_units_1, activation=args.activation))
    if(args.n_units_2 != -1):
        ann.add(tf.keras.layers.Dense(args.n_units_2, activation=args.activation))
    if(args.n_units_3 != -1):
        ann.add(tf.keras.layers.Dense(args.n_units_3, activation=args.activation))
    if(args.n_units_4 != -1):
        ann.add(tf.keras.layers.Dense(args.n_units_4, activation=args.activation))
    ann.add(tf.keras.layers.Dense(labels.shape[1]))
    #compile model
    ann.compile(loss=args.loss, optimizer=args.optimizer)
    #initialize metrics
    metrics_names = ["RMSE"]
    metrics = [tf.metrics.RootMeanSquaredError()]
    #determine if tuning or training final model
    if(args.cross_validation): #tuning using cross validation
        errors = util.cross_validation(features, labels, ann, args.n_splits, args.batch_size, args.n_epochs, metrics) #get the errors from the CV
        err_means = errors.mean(axis=0) #get means
        #output mean error to of each metric for hypertuning
        print(err_means)
        hpt = hypertune.HyperTune() 
        for i in range(len(metrics)):
            hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag=metrics_names[i], metric_value=err_means[i])
    else: #training for prediction
        ann.fit(features, labels, batch_size=args.batch_size, epochs=args.n_epochs) #fit the model using all the data
        ann.save(args.model_name + ".h5")
        util.save_model(args.model_dir, args.model_name + ".h5") #upload the saved model to the cloud
        
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
    parser.add_argument("--model-dir",
                            type=str,
                            default="badgerx-model-training",
                            help="Where to save the model")
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
    parser.add_argument("--batch-size",
                            type=int,
                            default=32,
                            help="size of each batch for training (default: 32)")
    parser.add_argument("--optimizer",
                            type=str,
                            choices=["adadelta", "adagrad", "adam", "adamax", "ftrl", "nadam", "rmsprop", "sgd"],
                            default="adam",
                            help="type of optimizer to use for compiling the model (defautl: adam)")
    parser.add_argument("--loss",
                            type=str,
                            choices=["mse", "mae", "mape", "msle"],
                            default="mse",
                            help="type of loss function to use (default: mse)")
    parser.add_argument("--n-epochs",
                            type=int,
                            default=1,
                            help="the number of epochs to train the model (default: 1)")
    parser.add_argument("--n-units-1",
                            type=int,
                            default=64,
                            help="list of number of units for the first ann layer (default: 64)")
    parser.add_argument("--n-units-2",
                            type=int,
                            default=-1,
                            help="list of number of units for the second ann layer (default: No second layer)")
    parser.add_argument("--n-units-3",
                            type=int,
                            default=-1,
                            help="list of number of units for the third ann layer (default: No second layer)")
    parser.add_argument("--n-units-4",
                            type=int,
                            default=-1,
                            help="list of number of units for the fourth ann layer (default: No third layer)")
    parser.add_argument("--activation",
                            type=str,
                            choices=["relu", "linear", "sigmoid", "tanh"],
                            default="relu",
                            help="list of number of units for the second rnn layer (default: No fourth layer)")
            
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    train_ann(args)

if __name__ == "__main__":
    main()
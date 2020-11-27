import numpy as np
import tensorflow as tf
import argparse
import util
import hypertune

def train_lr(args):
    #load data
    features, labels = util.load_data(args.features_file, args.labels_file)
    features = features.reshape(features.shape[0], args.input_shape_1, args.input_shape_2, args.input_shape_3)
    labels = labels.reshape(labels.shape[0], args.input_shape_1, args.input_shape_2)
    #convert time series data to supervised learning problem
    features, labels = util.convert_time_series_to_array(features, labels, args.input_width, args.input_stride, args.sampling_rate)
    features = features.astype(np.float32)
    labels = labels.astype(np.float32)
    #initialize model
    convlstm = tf.keras.models.Sequential([tf.keras.layers.ConvLSTM2D(filters=args.n_filters_1, activation='relu', kernel_size=args.kernel_size_1, padding='same', return_sequences=(args.n_filters_2 != -1))])
    if(args.n_filters_2 != -1):
        convlstm.add(tf.keras.layers.ConvLSTM2D(filters=args.n_filters_2, activation='relu', padding='same', kernel_size=args.kernel_size_2))
    convlstm.add(tf.keras.layers.Conv2D(filters=1, activation='relu', padding='same', kernel_size=args.kernel_size_3))
    #compile model
    convlstm.compile(loss=args.loss, optimizer=args.optimizer)
    #initialize metrics
    metrics_names = ["RMSE"]
    metrics = [tf.metrics.RootMeanSquaredError()]
    #determine if tuning or training final model
    if(args.cross_validation): #tuning using cross validation
        errors = util.cross_validation(features, labels, convlstm, args.n_splits, args.batch_size, args.n_epochs, metrics) #get the errors from the CV
        err_means = errors.mean(axis=0) #get means
        #output mean error to of each metric for hypertuning
        print(err_means)
        hpt = hypertune.HyperTune() 
        for i in range(len(metrics)):
            hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag=metrics_names[i], metric_value=err_means[i])
    else: #training for prediction
        convlstm.fit(features, labels, batch_size=args.batch_size, epochs=args.n_epochs) #fit the model using all the data
        convlstm.save(args.model_name + ".h5")
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
    parser.add_argument("--input-shape-1",
                            required=True,
                            type=int,
                            help="grid_X dimension of the input")
    parser.add_argument("--input-shape-2",
                            required=True,
                            type=int,
                            help="grid_Y dimension of the input")
    parser.add_argument("--input-shape-3",
                            required=True,
                            type=int,
                            help="num_featuresfeatures dimension of the input")
    parser.add_argument("--cross-validation",
                            type=str2bool, 
                            choices=["true", "false"],
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
    parser.add_argument("--n-filters-1",
                            type=int,
                            default=32,
                            help="list of number of filters for the first convlstm layer (default: 32)")
    parser.add_argument("--n-filters-2",
                            type=int,
                            default=-1,
                            help="list of number of filters for the second convlstm layer (default: no second layer)")
    parser.add_argument("--kernel-size-1",
                            type=int,
                            default=2,
                            help="kernel size for the first convlstm layer (default: 2)")
    parser.add_argument("--kernel-size-2",
                            type=int,
                            default=2,
                            help="kernel size for the second convlstm layer (default: 2)")
    parser.add_argument("--kernel-size-3",
                            type=int,
                            default=2,
                            help="kernel size for the final conv layer (default: 2)")
            
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    train_lr(args)

if __name__ == "__main__":
    main()
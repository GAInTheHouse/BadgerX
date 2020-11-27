import numpy as np
import pandas as pd 
import datetime
import tensorflow as tf
from google.cloud import storage
from sklearn.model_selection import KFold

def convert_time_series_to_array(data, labels, seq_length, seq_stride, sampling_rate):
    """Converts a time series dataset into a supervised learning problem"""
    X = []
    y = []
    for i in range(0, len(data) - seq_length, seq_stride): # i tracks the start of the sequence, start at 0 move by stride
        X.append(data[i:i + seq_length:sampling_rate]) # create sequences
        y.append(labels[i + seq_length: i + seq_length + 1])
    X = np.array(X) # shape is (samples, seq_length, features)
    y = np.array(y) 
    y = y.reshape(y.shape[0], y.shape[2], y.shape[3]) # shape is (samples, label_width, label_features) 
    return X, y # return arr as numpy array

def cross_validation(data, labels, model, n_splits, batch_size, epochs, metrics):
    """Performs time series walk-forward validation on data"""
    errors = [] # store all the errors
    tss = KFold(n_splits=n_splits) # make time series split object
    for train_idx, test_idx in tss.split(data, labels): # iterate through train test splits
        #train and validate models
        model.fit(data[train_idx], labels[train_idx], batch_size=batch_size, epochs=epochs)
        y_pred = model.predict(data[test_idx], batch_size=batch_size)
        metric_errors = [] # store errors based on each metric
        for metric in metrics: # iterate through metrics
            # MUST USE tf.metrics OR CHANGE CODE
            metric.reset_states() 
            metric.update_state(y_pred, labels[test_idx])
            metric_errors.append(metric.result().numpy())
        errors.append(metric_errors)
    return np.array(errors)

def load_data(features_file, labels_file):
    """Load data from google cloud based on provided filenames"""
    # Download the files
    # bucket = storage.Client(project="projectx-294502").bucket("badgerx-model-training")
    # features_blob = bucket.blob(features_file)
    # labels_blob = bucket.blob(labels_file) 
    # features_blob.download_to_filename("features.h5")
    # labels_blob.download_to_filename("labels.h5")

    # Read the downloaded hdf files
    features = pd.read_hdf("features.h5").values
    labels = pd.read_hdf("labels.h5").values

    return features, labels

def save_model(model_dir, model_name):
    """Saves the model to Google Cloud Storage"""
    bucket = storage.Client(project="projectx-294502").bucket(model_dir)
    blob = bucket.blob('trained_models/rnn/{}/{}'.format(
        datetime.datetime.now().strftime('model_%Y%m%d_%H%M%S'),
        model_name))
    blob.upload_from_filename(model_name)
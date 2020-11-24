import numpy as np
import pandas as pd 
import datetime
import tensorflow as tf
from google.cloud import storage
from sklearn.model_selection import TimeSeriesSplit

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def make_dataset(data, labels, input_width, input_stride, sampling_rate, batch_size):
    def split_window(features, input_width):
        inputs = features[:, :input_width, :, :, :]
        labels = features[:, input_width:, :, :, :]
        inputs = inputs[:, :, :, :, :-1]
        labels = labels[:, :, :, :, -1:]

        inputs.set_shape([None, input_width, None, None, None])
        labels = tf.reshape(labels, [tf.shape(features)[0], labels.shape[2], labels.shape[3], labels.shape[4]])
        return inputs, labels
    
    data = np.concatenate((data, labels[:, :, :, np.newaxis]), axis=3)

    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=input_width,
        sequence_stride=input_stride,
        sampling_rate=sampling_rate,
        shuffle=True,
        batch_size=batch_size)
    
    ds = ds.map(lambda x: split_window(x, input_width))
    
    return ds

def cross_validation(data, labels, model, n_splits, batch_size, epochs, metrics):
    """Performs time series walk-forward validation on data"""
    errors = [] # store all the errors
    tss = TimeSeriesSplit(n_splits=n_splits) # make time series split object
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
    bucket = storage.Client(project="projectx-294502").bucket("badgerx-model-training")
    features_blob = bucket.blob(features_file)
    labels_blob = bucket.blob(labels_file) 
    features_blob.download_to_filename("features.h5")
    labels_blob.download_to_filename("labels.h5")

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
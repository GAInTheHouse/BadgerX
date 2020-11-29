import numpy as np 
import pandas as pd
import joblib
import tensorflow as tf 

pca_tf_models = []
pca_sk_models = []
ae_tf_models = []
ae_sk_models = []

pca_test = pd.read_hdf('').values
ae_test = pd.read_hdf('').values 
labels_test = pd.read_hdf('').values

rmse = tf.metrics.RootMeanSquaredError()

for file in pca_tf_models:
    model = tf.keras.models.load_model(file)
    y_pred = model.predict(pca_test)
    rmse.update_state(y_pred, labels_test)
    print(file + ": " + str(rmse.result().numpy()))
    rmse.reset_states()
    y_pred = pd.DataFrame(y_pred)
    y_pred.to_hdf(file.split(".")[0] + "_pred.h5", "data", mode="w", format="f")

for file in ae_tf_models:
    model = tf.keras.models.load_model(file)
    y_pred = model.predict(ae_test)
    rmse.update_state(y_pred, labels_test)
    print(file + ": " + str(rmse.result().numpy()))
    rmse.reset_states()
    y_pred = pd.DataFrame(y_pred)
    y_pred.to_hdf(file.split(".")[0] + "_pred.h5", "data", mode="w", format="f")

for file in pca_sk_models:
    model = joblib.load(file)
    y_pred = model.predict(pca_test)
    rmse.update_state(y_pred, labels_test)
    print(file + ": " + str(rmse.result().numpy()))
    rmse.reset_states()
    y_pred = pd.DataFrame(y_pred)
    y_pred.to_hdf(file.split(".")[0] + "_pred.h5", "data", mode="w", format="f")

for file in ae_sk_models:
    model = joblib.load(file)
    y_pred = model.predict(ae_test)
    rmse.update_state(y_pred, labels_test)
    print(file + ": " + str(rmse.result().numpy()))
    rmse.reset_states()
    y_pred = pd.DataFrame(y_pred)
    y_pred.to_hdf(file.split(".")[0] + "_pred.h5", "data", mode="w", format="f")
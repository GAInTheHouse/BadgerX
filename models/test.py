import pandas as pd 
import numpy as np 

features = np.arange(200 * 15 * 15 * 3).reshape(200, 15 * 15 * 3)
labels = np.arange(200 * 15 * 15).reshape(200, 15 * 15)

features = pd.DataFrame(features)
labels = pd.DataFrame(labels)

features.to_hdf("features.h5", "data", format="f", mode="w")
labels.to_hdf("labels.h5", "data", format="f", mode="w")
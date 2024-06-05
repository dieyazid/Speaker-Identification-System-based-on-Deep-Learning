import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the training data
X_train = np.load('Assets\X_train.npy')
y_train = np.load('Assets\y_train.npy')
label_classes = np.load('Assets\label_classes.npy')

# Flatten the MFCCs and create a DataFrame
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
df = pd.DataFrame(X_train_flattened)

# Add the labels to the DataFrame
df['label'] = y_train

# Map label indices to label names
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes
df['label'] = label_encoder.inverse_transform(df['label'])

# # Save the DataFrame to a CSV file
# df.to_csv('training_data.csv', index=False)

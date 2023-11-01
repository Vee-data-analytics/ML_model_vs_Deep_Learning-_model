import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

prep_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/job_1.1.0.csv')

data = prep_df

# Define your features (X) and target variables (y)
X = data[['Polytec_RPM']]
y_strain_axial = data['Strain_Axial']
y_strain_torque = data['Strain_Torque']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train_strain_axial, y_test_strain_axial, y_train_strain_torque, y_test_strain_torque = train_test_split(
    X, y_strain_axial, y_strain_torque, test_size=0.2, random_state=42)

# Create separate Random Forest Regressor models for each target variable
rf_strain_axial = RandomForestRegressor(n_estimators=100, random_state=42)
rf_strain_torque = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the models to the training data
rf_strain_axial.fit(X_train, y_train_strain_axial)
rf_strain_torque.fit(X_train, y_train_strain_torque)

# Make predictions on the test data
y_pred_strain_axial = rf_strain_axial.predict(X_test)
y_pred_strain_torque = rf_strain_torque.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

# Evaluate the models
mse_strain_axial = mean_squared_error(y_test_strain_axial, y_pred_strain_axial)
mse_strain_torque = mean_squared_error(y_test_strain_torque, y_pred_strain_torque)
r2_strain_axial = r2_score(y_test_strain_axial, y_pred_strain_axial)
r2_strain_torque = r2_score(y_test_strain_torque, y_pred_strain_torque)

# Print the evaluation metrics
print(f'Mean Squared Error (Strain_Axial): {mse_strain_axial}')
print(f'R-squared (Strain_Axial): {r2_strain_axial}')
print(f'Mean Squared Error (Strain_Torque): {mse_strain_torque}')
print(f'R-squared (Strain_Torque): {r2_strain_torque}')



import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression

# Read the data using Polars
prep_df = pl.read_csv('/content/drive/MyDrive/Colab Notebooks/job_1.1.0.csv')

# Define the data
data = prep_df

# Define the bin size (90mm in your case)
bin_size = 90

# Calculate the number of subsets
max_distance = data['Distance_Bottom'].max()
num_subsets = int(np.ceil(max_distance / bin_size))

# Initialize an empty dictionary to store the subsets
data_subsets = {}

# Create subsets based on 'Distance_Bottom'
for i in range(num_subsets):
    start = i * bin_size
    end = (i + 1) * bin_size
    subset_name = f'Subset_{i + 1}'
    data_subsets[subset_name] = data.filter((pl.col('Distance_Bottom') >= start) & (pl.col('Distance_Bottom') < end))

# Initialize dictionaries for predictions
predictions_strain_torque = {}
predictions_strain_axial = {}

# Iterate through the subsets and train a model for each
for subset_name, subset in data_subsets.items():
    X = subset['Polytec_RPM'].to_pandas().values.reshape(-1, 1)  # Independent variable
    y_torque = subset['Strain_Torque'].to_pandas().values  # Dependent variable (Strain_Torque)
    y_axial = subset['Strain_Axial'].to_pandas().values  # Dependent variable (Strain_Axial)

    # Initialize and train the model for Strain_Torque
    model_strain_torque = LinearRegression()
    model_strain_torque.fit(X, y_torque)

    # Initialize and train the model for Strain_Axial
    model_strain_axial = LinearRegression()
    model_strain_axial.fit(X, y_axial)

    # Now, you can use these models to predict Strain_Torque and Strain_Axial based on Polytec_RPM
    predictions_strain_torque[subset_name] = model_strain_torque.predict(X)
    predictions_strain_axial[subset_name] = model_strain_axial.predict(X)
 for subset_name, predictions in predictions_strain_torque.items():
     print(f"Subset: {subset_name}")
     print("Polytec RPM\tPredicted Strain Torque")
  for rpm, torque in zip(data_subsets[subset_name]['Polytec_RPM'], predictions):
      print(f"{rpm:.2f}\t{torque:.2f}")
      print()import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Read the data using Polars
prep_df = pl.read_csv('/content/drive/MyDrive/Colab Notebooks/job_1.1.0.csv')

# Define the data
data = prep_df

# Define the features (Bonfigloli_Power, Acc_Z, and Polytec_RPM)
features = ['Bonfigloli_Power', 'Acc_Z', 'Polytec_RPM']

# Define the bin size (90mm in your case)
bin_size = 90

# Calculate the number of subsets
max_distance = data['Distance_Bottom'].max()
num_subsets = int(np.ceil(max_distance / bin_size))

# Initialize dictionaries to store the models and performance metrics
models_torque = {}
models_axial = {}
performance_metrics_torque = {}
performance_metrics_axial = {}

# Iterate through subsets and train models
for i in range(num_subsets):
    subset_name = f'Subset_{i + 1}'

    # Create the subset based on 'Distance_Bottom'
    start = i * bin_size
    end = (i + 1) * bin_size
    subset = data.filter((pl.col('Distance_Bottom') >= start) & (pl.col('Distance_Bottom') < end))

    # Select the data for this subset
    X = subset[features].to_pandas().values
    y_torque = subset['Strain_Torque'].to_pandas().values
    y_axial = subset['Strain_Axial'].to_pandas().values

    # Split the data into training and testing sets
    X_train, X_test, y_train_torque, y_test_torque, y_train_axial, y_test_axial = train_test_split(
        X, y_torque, y_axial, test_size=0.2, random_state=42)

    # Initialize and train the model for Strain Torque
    model_torque = RandomForestRegressor(n_estimators=100, random_state=42)
    model_torque.fit(X_train, y_train_torque)

    # Initialize and train the model for Strain Axial
    model_axial = RandomForestRegressor(n_estimators=100, random_state=42)
    model_axial.fit(X_train, y_train_axial)

    # Predict Strain Torque and Strain Axial for the test set
    predictions_torque = model_torque.predict(X_test)
    predictions_axial = model_axial.predict(X_test)

    # Calculate and store the performance metrics
    mse_torque = mean_squared_error(y_test_torque, predictions_torque)
    r2_torque = r2_score(y_test_torque, predictions_torque)
    mse_axial = mean_squared_error(y_test_axial, predictions_axial)
    r2_axial = r2_score(y_test_axial, predictions_axial)

    # Store models and performance metrics
    models_torque[subset_name] = model_torque
    models_axial[subset_name] = model_axial
    performance_metrics_torque[subset_name] = {'MSE': mse_torque, 'R2': r2_torque}
    performance_metrics_axial[subset_name] = {'MSE': mse_axial, 'R2': r2_axial}

    # Print performance metrics for this subset
    print(f'Subset: {subset_name}')
    print(f'MSE for Strain Torque: {mse_torque:.2f}')
    print(f'R-squared for Strain Torque: {r2_torque:.2f}')
    print(f'MSE for Strain Axial: {mse_axial:.2f}')
    print(f'R-squared for Strain Axial: {r2_axial:.2f}')
    print('\n')

import polars as pl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Read the data using Polars
prep_df = pl.read_csv('/content/drive/MyDrive/Colab Notebooks/job_1.1.0.csv')

# Define the data
data = prep_df

# Define the features (Bonfigloli_Power, Acc_Z, and Polytec_RPM)
features = ['Bonfigloli_Power', 'Acc_Z', 'Polytec_RPM']

# Define the bin size (90mm in your case)
bin_size = 90

# Calculate the number of subsets
max_distance = data['Distance_Bottom'].max()
num_subsets = int(np.ceil(max_distance / bin_size))

# Initialize dictionaries to store the models and performance metrics
models_torque = {}
models_axial = {}
performance_metrics_torque = {}
performance_metrics_axial = {}

# Initialize a list to store model history (for deep learning models)
model_histories_torque = {}
model_histories_axial = {}

# Define deep learning model architecture
def create_deep_learning_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

# Iterate through subsets and train deep learning models
for i in range(num_subsets):
    subset_name = f'Subset_{i + 1}'

    # Create the subset based on 'Distance_Bottom'
    start = i * bin_size
    end = (i + 1) * bin_size
    subset = data.filter((pl.col('Distance_Bottom') >= start) & (pl.col('Distance_Bottom') < end))

    # Select the data for this subset
    X = subset[features].to_pandas().values
    y_torque = subset['Strain_Torque'].to_pandas().values
    y_axial = subset['Strain_Axial'].to_pandas().values

    # Split the data into training and testing sets
    X_train, X_test, y_train_torque, y_test_torque, y_train_axial, y_test_axial = train_test_split(
        X, y_torque, y_axial, test_size=0.2, random_state=42)

    # Initialize deep learning models for Strain Torque and Strain Axial
    model_torque = create_deep_learning_model(input_shape=(X_train.shape[1],))
    model_axial = create_deep_learning_model(input_shape=(X_train.shape[1],))

    # Train deep learning models for Strain Torque and Strain Axial
    history_torque = model_torque.fit(X_train, y_train_torque, validation_data=(X_test, y_test_torque), epochs=50, verbose=0)
    history_axial = model_axial.fit(X_train, y_train_axial, validation_data=(X_test, y_test_axial), epochs=50, verbose=0)

    # Predict Strain Torque and Strain Axial for the test set
    predictions_torque = model_torque.predict(X_test)
    predictions_axial = model_axial.predict(X_test)

    # Calculate and store the performance metrics for Strain Torque
    mse_torque = mean_squared_error(y_test_torque, predictions_torque)
    r2_torque = r2_score(y_test_torque, predictions_torque)

    # Calculate and store the performance metrics for Strain Axial
    mse_axial = mean_squared_error(y_test_axial, predictions_axial)
    r2_axial = r2_score(y_test_axial, predictions_axial)

    # Store models and performance metrics for Strain Torque
    models_torque[subset_name] = model_torque
    performance_metrics_torque[subset_name] = {'MSE': mse_torque, 'R2': r2_torque}
    model_histories_torque[subset_name] = history_torque

    # Store models and performance metrics for Strain Axial
    models_axial[subset_name] = model_axial
    performance_metrics_axial[subset_name] = {'MSE': mse_axial, 'R2': r2_axial}
    model_histories_axial[subset_name] = history_axial

    # Visualize model history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_torque.history['mean_squared_error'], label='Train MSE')
    plt.plot(history_torque.history['val_mean_squared_error'], label='Val MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.title(f'Deep Learning Model Training - Strain Torque - {subset_name}')

    plt.subplot(1, 2, 2)
    plt.plot(history_axial.history['mean_squared_error'], label='Train MSE')
    plt.plot(history_axial.history['val_mean_squared_error'], label='Val MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.title(f'Deep Learning Model Training - Strain Axial - {subset_name}')

    plt.tight_layout()
    plt.show()


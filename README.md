markdown

# Regression Analysis for Strain Axial and Strain Torque

This analysis involves building regression models to predict Strain Axial and Strain Torque based on the Polytec RPM. The dataset used for this analysis is loaded from a CSV file.

## Preparing the Data

First, the dataset is loaded and divided into features (X) and target variables (y). In this case, the Polytec RPM is the independent variable (X), and Strain Axial and Strain Torque are the dependent variables (y).

```python
# Define your features (X) and target variables (y)
X = data[['Polytec_RPM']]
y_strain_axial = data['Strain_Axial']
y_strain_torque = data['Strain_Torque']

The data is then split into training and testing sets, with 80% used for training and 20% for testing.
Building Random Forest Regression Models

Two separate Random Forest Regressor models are created, one for Strain Axial and another for Strain Torque. Each model is trained using the training data.

python

# Create separate Random Forest Regressor models for each target variable
rf_strain_axial = RandomForestRegressor(n_estimators=100, random_state=42)
rf_strain_torque = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the models to the training data
rf_strain_axial.fit(X_train, y_train_strain_axial)
rf_strain_torque fit(X_train, y_train_strain_torque)

Making Predictions

The models are used to make predictions on the test data.

python

# Make predictions on the test data
y_pred_strain_axial = rf_strain_axial.predict(X_test)
y_pred_strain_torque = rf_strain_torque.predict(X_test)

Evaluating the Models

The models' performance is evaluated using mean squared error (MSE) and R-squared (R²) metrics.

python

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

This analysis provides insights into the predictive performance of the Random Forest Regression models for Strain Axial and Strain Torque based on the Polytec RPM.
Deep Learning Regression Analysis

This analysis involves using deep learning models to predict Strain Torque and Strain Axial based on the features Bonfigloli_Power, Acc_Z, and Polytec_RPM. The dataset is divided into subsets based on the Distance_Bottom feature.
Data Preparation

The data is loaded using the Polars library, and the features (independent variables) and target variables (Strain Torque and Strain Axial) are defined.

python

# Define the features (Bonfigloli_Power, Acc_Z, and Polytec_RPM)
features = ['Bonfigloli_Power', 'Acc_Z', 'Polytec_RPM']

The data is then split into subsets based on the Distance_Bottom feature.
Training Deep Learning Models

Deep learning models are created and trained for each subset of data. Each model is trained separately for Strain Torque and Strain Axial.
Model Evaluation

The performance of the deep learning models is evaluated using mean squared error (MSE) and R-squared (R²) metrics. The models and their performance metrics are stored for each subset.
Model Training History

The training history of the deep learning models, including training MSE and validation MSE, is visualized for each subset.

In this analysis, deep learning models are applied to predict Strain Torque and Strain Axial, taking into account subsets of the data based on the Distance_Bottom feature. The training process and model performance are assessed, providing insights into the effectiveness of deep learning in regression tasks.

Please ensure you have the necessary libraries and data before running this analysis.

css


This Markdown file now includes both the Random Forest Regression analysis and the Deep Learning Regression analysis in one document. You can save this content to a `.md` file for documentation and sharing.


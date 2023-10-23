# Supply chain management project

## Team members

- SE20UARI144: Siddarth Reddy Gurram
- SE20UARI121: R Souri Satya Saketh
- SE20UARI176: Neha Tummala
- SE20UARI063: Hemal Tummapudi
- SE20UARI084: Kumbagiri Siddartha Rahul Reddy

## Libraries and Modules Used

- [Pandas](https://pandas.pydata.org/): A powerful data manipulation and analysis library.
- [NumPy](https://numpy.org/): Library for numerical computing in Python.
- [Matplotlib](https://matplotlib.org/): A data visualization library.
- [Seaborn](https://seaborn.pydata.org/): Data visualization library based on Matplotlib.
- [Scikit-learn](https://scikit-learn.org/stable/): Machine learning library for Python.
- [XGBoost](https://xgboost.readthedocs.io/): An efficient and scalable gradient boosting library.
- [LightGBM](https://lightgbm.readthedocs.io/): A gradient boosting framework that uses tree-based learning algorithms.

## Data Pre-processing

-Viewing the First Rows of a DataFrame

In data analysis and data science, it's often essential to examine the first few rows of the dataset to get a quick overview of its contents. In Python, one can easily achieve this using the `.head()` method, typically applied to a DataFrame.

-Checking for Missing Values

One of the critical tasks is to identify and handle missing values in the dataset. In Python, we can check for missing values using the `.isna()` method, which returns a DataFrame of the same shape as your original data, but with Boolean values indicating whether each element is missing.

-Handling Missing Values in the 'total_price' Column

In the process of data preprocessing, one task is dealing with missing values in the dataset. Missing values can impact the quality and accuracy of analysis and modeling. In this case, we are addressing missing values in the 'total_price' column of the dataset.

## Custom Functions for Feature Generation

In this project, we have defined several custom functions to generate new features based on the data. Here's an overview of these functions:

### `gen_count_id(train, test, col, name)`

This function generates a new feature that counts the occurrences of a specific column in the dataset. The parameters include:
- `train`: The training dataset.
- `test`: The test dataset.
- `col`: The column you want to count occurrences of.
- `name`: The name of the new feature.

### `gen_average_units(train, test, col, name)`

This function calculates the average of the 'units_sold' column within specific groups and generates a new feature. The parameters include:
- `train`: The training dataset.
- `test`: The test dataset.
- `col`: The column used for grouping.
- `name`: The name of the new feature.

### `gen_average_price(train, test, col, price, name)`

This function calculates the average of a specified price column within specific groups and generates a new feature. The parameters include:
- `train`: The training dataset.
- `test`: The test dataset.
- `col`: The column used for grouping.
- `price`: The price column to calculate the average for.
- `name`: The name of the new feature.

These custom functions are helpful for creating additional features based on the dataset's characteristics and can improve the performance of the machine learning models. 

We are performing feature generation and transformation to enhance our dataset for better model performance. Here's an overview of the steps taken:

### Counting Records per 'sku-id & store-id'

We are counting the number of records per 'sku-id & store-id,' 'sku-id,' and 'store-id' and creating new features 'count_id_sku_store,' 'count_id_sku,' and 'count_id_store,' respectively.

### Average Units Sold per Group

We calculate the average 'units_sold' per 'sku-id & store-id,' 'store-id,' and 'sku-id' and generate new features 'count_sku_store_id,' 'count_store_id,' and 'count_sku_id,' respectively.

### Average Price per Group

We compute the average 'base_price' and 'total_price' per 'sku-id & store-id,' 'store-id,' and 'sku-id.' The resulting features are 'price_sku_store,' 'price_to_sku_store,' 'price_store_id,' 'price_sku_id,' 'price_to_store_id,' and 'price_to_sku_id.'

### Week Transformation

We transform the 'week' feature using various techniques:
- We encode 'week' using an Ordinal Encoder to create a new feature 'week_1.'
- We create 'week_num' and 'week_num1' features by calculating the week number and a modulo 4 value.
- We encode 'week_1' as a cyclic feature using sine and cosine transforms, resulting in 'week_sin' and 'week_cos' features.

### Price Difference Percentage

We calculate the percentage difference between 'base_price' and 'total_price' and create the 'price_diff_percent' feature.

These feature engineering and transformation steps help prepare the data for modeling and can lead to improved prediction accuracy. Make sure to adjust the parameters and function calls according to your specific dataset and project requirements.

## Data Visualization
## Automated Visualization using AutoViz
-In this section, we utilize the AutoViz_Class from the autoviz library to generate automated visualizations for the dataset.
-When the code is run, it will generate and display a variety of visualizations based on the dataset. These visualizations include histograms, scatter plots, bar charts, and correlation matrices providing an overview of the data's characteristics.
-Dependencies
autoviz: Need to install the autoviz library using !pip install autoviz before running this code.
References
autoviz library documentation: https://pypi.org/project/autoviz/

## Training and Testing Data
-In this section, we describe the code used for selecting the features (predictors) and the target variable (response) for your predictive modeling task.
-X: This variable contains the training dataset with a subset of columns. Features like 'record_ID' and 'week' are excluded to ensure the model focuses on relevant attributes.
-X_test: Similarly, this variable contains the testing dataset with the same columns excluded.

### Target Variable
-The code also deals with the target variable, transforming it using the log1p function:
-Y: This variable represents the transformed target variable, which can be used in modeling. Using the natural logarithm transformation can help to stabilize variance and improve the modeling performance.
### Purpose:
-Feature selection is crucial to ensure that the machine learning models have relevant input data for predicting the target variable. By excluding unnecessary or irrelevant columns, we can improve the efficiency and accuracy of your models.
-Transforming the target variable can be helpful when it exhibits non-linear patterns or significant skewness. 
### Usage
-We can apply these feature selection and target variable transformation steps to the train and test datasets as shown in the provided code. The excluded columns can be adjusted based on your specific modeling requirements.
-The transformed target variable Y can be used as the response variable when training and evaluating your predictive models.

### Categorical Variable Encoding using M-Estimate Encoder
In this section, we try to explain the code used for encoding categorical variables using the M-Estimate Encoder.

### M-Estimate Encoder
-M-Estimate Encoding is a technique used to transform categorical variables into numerical format, which is essential for most machine learning models. 

## Training a RandomForestRegressor
-In this section, we describe the code used for training a RandomForestRegressor, which is a powerful ensemble machine learning algorithm for regression tasks.
-RandomForestRegressor is an ensemble learning method that combines the predictions from multiple decision tree regressors to make accurate and robust predictions. It is useful for a wide range of regression problems.
-rf_base: This variable represents an instance of the RandomForestRegressor.
-rf_base.fit(x_train, y_train): This step fits the regressor to the training data, where x_train is the training data, and y_train is the corresponding target variable.
### Purpose:
-The purpose of training a RandomForestRegressor is to build a predictive model for regression tasks. It works by constructing multiple decision trees and averaging their predictions, which typically results in a more accurate and stable model.

## Training LightGBM Regressors
-In this section, we describe the code used for training LightGBM regressors, both a base and a tuned model.
-LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is highly efficient and effective for regression tasks and it can be tuned for better performance.
-model_lgb_tuned: This variable represents an instance of the LightGBM Regressor with tuned hyperparameters.
-x_train and y_train: These variables are the preprocessed training data and the target variable.
### Purpose:
-The purpose of training LightGBM regressors is to build predictive models for regression tasks. LightGBM is known for its efficiency and high predictive performance. The base model provides a starting point, while the tuned model incorporates specific hyperparameters to optimize performance.

### Hyperparameter Tuning:
The tuned model is configured with specific hyperparameters that have been fine-tuned to potentially improve predictive accuracy. We can further adjust these hyperparameters or use techniques like grid search or random search to find the best hyperparameters for the specific dataset and regression task.

## Generating Predictions and Calculating MSLE
-In this section, we describe the code used for generating predictions and calculating Mean Squared Log Error (MSLE) for Random Forest (RF) and LightGBM (LGBM) models. Additionally, an ensemble approach is used to combine predictions from both models.

### Purpose:
-The purpose of this code is to assess and compare the performance of two machine learning models, RF and LGBM, on a validation dataset and then create an ensemble prediction that combines the strengths of both models.

### Usage
-prediction_rfb_valid: Predictions for the validation dataset using the base Random Forest model.
-prediction_rft_valid: Predictions for the validation dataset using the tuned Random Forest model.
-prediction_lgbmb_valid: Predictions for the validation dataset using the base LightGBM model.
-prediction_lgbmt_valid: Predictions for the validation dataset using the tuned LightGBM model.
-rf_base_msle: MSLE for the base Random Forest model's predictions.
-rf_tuned_msle: MSLE for the tuned Random Forest model's predictions.
-lgbm_base_msle: MSLE for the base LightGBM model's predictions.
-lgbm_tuned_msle: MSLE for the tuned LightGBM model's predictions.
-prediction_ensemble_base: Ensemble predictions for the base models that combine the strengths of RF and LGBM.
-prediction_ensemble_tuned: Ensemble predictions for the tuned models that combine the strengths of RF and LGBM.
-ensemble_base_msle: MSLE for the ensemble predictions based on the base models.
-ensemble_tuned_msle: MSLE for the ensemble predictions based on the tuned models.
### Purpose of Ensemble
-The ensemble predictions are intended to combine the predictions from different models, which can help improve prediction accuracy by leveraging the strengths of both models. The weights in the ensemble are determined based on each model's performance.

### Interpretation of MSLE
-MSLE is a metric for regression tasks that quantifies the mean squared logarithmic error between predicted and actual values. It is used to evaluate the quality of regression models, where lower MSLE values indicate better model performance.

## Creating an output_df DataFrame
-In this section, we describe the code used to create a DataFrame named output_df. This DataFrame is typically used to store and organize the results or predictions for a specific task.

### Purpose:
-The purpose of this code is to prepare a structured DataFrame, output_df, that will store specific data, such as the 'record_ID,' which is often used for identifying records or observations in a dataset.
-test3: This is the source DataFrame that contains the data from which you want to extract the 'record_ID' column.
-[['record_ID']]: This part of the code specifies which columns from the 'test3' DataFrame should be included in the output_df. In this case, it selects only the 'record_ID' column.
-.copy(): This method creates a copy of the selected columns, ensuring that changes made to output_df do not affect the original DataFrame.
## Finding the Minimum Value in the 'units_sold' Column
-In this section, we describe the code used to find the minimum value in the 'units_sold' column of the train DataFrame.

### Purpose:
-The purpose of this code is to determine the minimum value of the 'units_sold' column within the train DataFrame. This can provide insights into the dataset and help identify the smallest number of units sold among the recorded observations.
-train: This is the source DataFrame from which you want to extract the minimum value.
-['units_sold']: This part of the code specifies the column 'units_sold,' which you want to analyze.
-.min(): This method is used to calculate the minimum value within the specified column.

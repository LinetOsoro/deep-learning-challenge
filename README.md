# Deep Learning 
## Preprocess the Data

Upload the starter file to Google Colab.

Read in the charity_data.csv to a Pandas DataFrame.

Identify the target variable(s) and feature(s) for the model.

Drop the EIN and NAME columns.

Determine the number of unique values for each column.

Use pd.get_dummies() to encode categorical variables.

Split the preprocessed data into features array, X, and a target array, y.

Scale the training and testing features datasets using StandardScaler.


## Compile, Train, and Evaluate the Model


Designed a neural network model with  input features and nodes for each layer

Created hidden layers with appropriate activation functions.

Created an output layer with an appropriate activation function.


##  Optimize the Model

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Repeat the preprocessing steps in a new Jupyter notebook

Create a new neural network model, implementing at least 3 model optimization methods 

Save and export your results to an HDF5 file named AlphabetSoupCharity_Optimization.h5


# Analysis Report
Report on the Performance of Deep Learning Models for Alphabet Soup


## Overview
The objective of the deep learning model is to select funding applicants with the highest probability of success, aiming for
an accuracy rate of 75% or higher.


## Data Preprocessing
Target Variable: 'IS_SUCCESSFUL' column from application_df.
    
Feature Variables: 'SPECIAL_CONSIDERATIONS', 'APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE', 
                        'ORGANIZATION','INCOME_AMT'.
        
Removed Variables: 'EIN' and 'NAME' columns due to their lack of relevance.

    
## Model Architecture and Performance

| Model Attempt                              | Model Parameters                                            | Accuracy | Loss  |
|--------------------------------------------|-------------------------------------------------------------|----------|-------|
| Attempt 1 (AlphabetSoupCharity)            | 80 nodes in layer 1, 30 nodes in layer 2                   | 73.24%   | 55.06%|
| Attempt 2 (AlphabetSoupCharity2_Optimization) | 40 nodes in layer 1, 15 nodes in layer 2                | 72.26%   | 55.81%|
| Attempt 3 (AlphabetSoupCharity1_Optimization) | 12 nodes in layer 1, 8 nodes in layer 2, 4 nodes in layer 3 | 73.24%   | 55.46%|


#### Attempt 1:(AlphabetSoupCharity): Model Parameters: 80 hidden nodes in layer 1, 30 hidden nodes in layer 2.
Result: Accuracy: 73.24%, Loss: 55.06%.

#### Attempt 2:(AlphabetSoupCharity2_Optimization)
Model Parameters: Reduced hidden nodes to half of Attempt 1 (40 in layer 1, 15 in layer 2).
Result: Accuracy: 72.26%, Loss: 55.81%.

#### Attempt 3:(AlphabetSoupCharity1_Optimization)
Model Parameters: Implemented 3 layers, used ReLU activation in the second layer, with 12 hidden nodes in layer 1, 8 in layer 2, and 4 in layer 3.Result: Accuracy: 73.24%, Loss: 55.46%.
            

## Achievement of Target Performance
The target accuracy of 75% was not achieved with any of the attempts.


## Steps for Performance Improvement
To enhance model performance, the following steps were taken:

Increased the number of layers.

Added additional hidden nodes.

Adjusted activation functions.


## Summary
The deep learning model achieved an accuracy of approximately 73% in predicting successful applicants for funding. 
To further improve accuracy, additional data cleanup and experimentation with different model architectures and 
activation functions are recommended.




## Resources used 
[How to Save to HDF5 file](https://stackoverflow.com/questions/43402320/export-tensorflow-weights-to-hdf5-file-and-model-to-keras-model-json)

[TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/Model)

[Keras](https://keras.io/keras_3/)

# Deep Learning Challenge 

## Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures.  With the knowledge of machine learning and neural networks, and using the features in the provided dataset (mentioned below), a binary classifier was created that can predict whether applicants will be successful if funded by Alphabet Soup.

Alphabet Soup's business team provided a CSV dataset, charity_data.csv, that containd information about 34,299 organizations that have received funding from Alphabet Soup over the years.  Within that dataset were a number of columns that captured metadata about each organization.  The columns included the following information:
   - EIN and NAME = The 2 identification columns.
   - APPLICATION_TYPE = Alphabet Soup application type.
   - AFFILIATION = Affiliation sector of industry.
   - CLASSIFICATION = Government organization classification.
   - USE_CASE = Use case for funding.
   - ORGANIZATION = Organization type.
   - STATUS = Active status.
   - INCOME_AMT = Income classification.
   - SPECIAL_CONSIDERATIONS = Special considerations for application.
   - ASK_AMT = Funding amount requested.
   - IS-SUCCESSFUL = Was the money used effectively.

### Before opening the StarterCode folder

1. I created a new repository in GitHub for this project called `deep-learning-challenge`. 
2. Inside the new repository I cloned the new repository to my computer.
3. Inside my local Git repository, I added the starter Jupitor Notebook code Starter_Code.ipynb.
4. I uploaded Starter_Code.ipynb into Google Colab and completed the steps for building the first Neural Network model and the second optimized Neural Network model.

## Overview of the Analysis

1. The dataset charity_data.csv contained potential features (variables) about the 34,299 organizations that requested funding from the non-profit foundation Alphabet Soup. 18,261 of the organizations were identified as "using the funds effectively" (IS-SUCCESSFUL = 1) and the remaining 16,038 organizations were identified as "not using the funds effectively" (IS-SUCCESSFUL = 0).  The purpose of the analysis is to build a Machine Learning model that would predict the classification of organizations into "successful" vs. "not successful" in using the funds effectively; that is, predict whether applicants will be successful if funded by Alphabet Soup.

2. The following steps were implemented for building both Neural Network models:
   - STEP 1: Pre-processed the data.
      * Read in the charity_data.csv to a Pandas DataFrame, and identified the target and the features for the model.
      * Dropped columns that were not considered features for the model.
      * Determined the number of unique values for each column.
      * Used the number of data points for each unique value to pick a cutoff point to bin "rare" categorial variables together in a new value "other", and checked if the           binning was successful.
      * Used pd.get.dummies() to encode categorical variables.
      * Split the preprocessed data into a features array x, and a target array y.  Then used these arrays and the "train_test_split" function to split the data into                training and testing datasets.
      * Scaled the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then used the "transform" function.     
   - STEP 2: Compiled, Trained, and Evaluated the Model(s).
      * Created a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
      * Created the first hidden layer and chose an appropriate activation function.
      * Added a second hiddne layer, if necessary, with an appropriate activation function.
      * Checked the structure of the model.
      * Compiled and trained the model.
      * Determined the number of Epochs.
      * Evaluated the model using the test data to determine the loss and accuracy.
      * Saved and exported the model results to an HDF5 file.
   - STEP 3: Optimized the first Neural Network model to achieve a target predictive accuracy higher than 75%.
     
3. Two different models were developed:
   - Neural Network Model #1
     * Target Variable = "IS_SUCCESSFUL"
     * Features = "APPLICATION_TYPE","AFFILIATION","CLASSIFICATION","USE_CASE","ORGANIZATION","STATUS","INCOME_AMT",
                   "SPECIAL_CONSIDERATIONS","ASK_AMT".
     * Neither Target nor Features = "EIN" and "NAME", the 2 identification columns, that were removed from the input data.
     * 2 hidden layers and 1 output layer.  Given the few # of features for consideration, the almost 50/50 split between successful vs. non-successful organizations (53%          and 47%, respectively), I anticipated that 2 hidden layers and the selected activation functions would result in an acceptable accuracy and loss scores.
          * The first hidden layer had 10 nodes, and activation function "tanh", 440 parameters.
          * The second hidden layer had 5 nodes, and activation function "relu", 55 parameters.
          * The ouput layer had 1 node, and activation function "relu", 6 parameters.
          * Ran 100 epochs.
     * After 100 epochs, including binning and scaling the training and testing features' datasets, the model's accuracy score was 72.7% and the loss score was 57.5%.
      
   - (Optimized) Neural Network Model #2
     * Target Variable = "IS_SUCCESSFUL"
     * Features = "NAME","APPLICATION_TYPE","AFFILIATION","CLASSIFICATION","USE_CASE","ORGANIZATION","STATUS","INCOME_AMT",
                   "SPECIAL_CONSIDERATIONS","ASK_AMT".
       (Note-added NAME as a feature, and implemented binning for NAME catergories with counts less than 100 into "Other" category)
     * Neither Target nor Features = "EIN", the 1 identification column, that was removed from the input data.
     * 3 hidden layers and 1 output layer.  Given that the first Neural Network model did not achieve the target model performance of predictive accuracy score of more than        75%, I anticipated that adding another hidden layer, increasing the number of nodes for the second hidden layer, and changing activation function to sigmoid could           help in improving the accuracy of the Neural Network Model. 
          * The first hidden layer had 10 nodes, and activation function "relu", 750 parameters.
          * The second hidden layer had 8 nodes, and activation function "sigmoid", 88 parameters.
          * The third hidden layer had 6 nodes, and activation function "sigmoid", 54 paramters.
          * The ouput layer had 1 node, and activation function "relu", 7 parameters.
          * Ran 50 epochs.
     * After only 50 epochs (half the number of epochs for the first Neural Network model, including binning and scaling the training and testing features' datasets, the           model's accuracy score was 75.3% and the loss score was 49.2%.  Thus, achieving the target model performance of a predictive accuracy score of more than 75%.

     
## Consideration for further improvements to the model's performance

Further attempts to improve the predictive accuracy score of future Neural Network models might be achieved by exploring the correlation between the different features and the target variable, to include those features with the highest correlation coefficients in the predictive model.  Another consideration could be using Principal Component Analysis (PCA) for reducing the number of features instead of scaling the features using "StandardScaler."

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

### Before opening the starterCode folder

1. I created a new repository in GitHub for this project called `deep-learning-challenge`. 
2. Inside the new repository I cloned the new repository to my computer.
3. Inside my local Git repository, I added the starter Jupitor Notebook code Starter_Code.ipynb.
4. I uploaded Starter_Code.ipynb into Google Colab and completed the steps for building the first Neural Network model and the second optimized Neural Network model.

## Overview of the Analysis

1. The dataset charity_data.csv contained potential features (variables) about the 34,299 organizations that requested funding from the non-profit foundation Alphabet Soup. 18,261 of the organizations are identified as "using the funds effectively" (IS-SUCCESSFUL = 1) and the remaining 16,038 organizations are identified as "not using the funds effectively" (IS-SUCCESSFUL = 0).  The purpose of the analysis is to build a neural network model that would predict the classification of organizations into "successful" vs. "not successful" in using the funds effectively; that is, predict whether applicants will be successful if funded by Alphabet Soup.

2. The following steps were implemented for building both Neural Network models:
   - Pre-proceed the data.
      * Read in the charity_data.csv to a Pandas DataFrame, and identified the target and the features for the model.
      * Dropped columns that were not considered features for the model.
      * Determined the number of unique values for each column.
      * Used the number of data points for each unique value to pick a cutoff point to bin "rare" categorial variables together in a new value "other", and checked if the           binning was successful.
      * Used pd.get.dummies() to encode categorical variables.
      * Split the preprocessed data into a features array x, and a target array y.  Then used these arrays and the train_test_split function to split the data into training         and testing datasets.
      * Scaled the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then used the transform function.
        
   - Save the predictions for the testing data labels by using the testing feature data (x_test) and the fitted model.
   - Evaluate the model's performance by generating a confusion matrix and printing the classification report.

3. Two different models were developed:
   - Logistic Regression Model (#1) with the original dataset lending_data.csv, that included 75,036 low-risk loans and 2,500 high-risk loans.  The dataset was split into 2      groups (75% to 25% split); the training dataset for building the model with 58,152 borrowers that included 56,277 low-risk loans and 1,875 high-risk loans, and the          test dataset with 19,384 borrowers that included 18,759 low-risk loans and 625 high-risk loans.
   - Logistics Regression Model (#2) with oversampling the high-risk loans in the training dataset using RandomOverSampler module from imbalanced-learn, which generated a        new dataset of 56,277 low-risk loans (same as what was included in the Logistic Regression Model #1 training dataset) and 56,277 high-risk loans (oversampling the # of      high-risk loans from the original 1,875 that was included in the Logistic Regrsssion #1 training dataset).

4. The following steps were implemented for building both Logistics Regression models:
   - Fit a logistic regression model by using the training dataset (x_train and y_train).
   - Save the predictions for the testing data labels by using the testing feature data (x_test) and the fitted model.
   - Evaluate the model's performance by generating a confusion matrix and printing the classification report.

5. Logistic Regression Model (#1) Performance Results:
   - Precision in predicting low-risk loans = 100%.  Precision in predicting high-risk loans = 87%.
   - Accuracy = 94.4%
   - Recall in predicting low-risk loans = 100%.  Recall in predicting high-risk loans = 89%.
  
   Logistic Regression Model (#2) Performance Results:
   - Precision in predicting low-risk loans = 100%.  Precision in predicting high-risk loans = 87%.
   - Accuracy = 99.6%
   - Recall in predicting low-risk loans = 100%.  Recall in predicting high-risk loans = 100%.
     
## Overview of the Prediction Analysis

The Logistic Regression Model (#1) predicts a healthy (low-risk) loan with 100% precision, while it predicts a high-risk loan with a lower precision at 87%. In general,  that logistic regression model is good at predicting whether a loan may default (not a healthy loan, or is a high risk loan) because of its high balanced accuracy at 94.4% and somewhat high f-1 and recall scores. If the bank is still getting a high precision and recall on the test dataset (even if they are lower scores than for the training dataset), it is a good indication about how well the model is likely to perform in real life.  Consequently, the accuracy of the logistic regression model seems to be good enough to start exploring this algorithm in a bank setting for assessing the creditworthiness of borrowers; however, it may be prudent for the bank to start running a pilot with new data to assess the model's reliability on data the model has not "seen" yet.   

The resampled Logistic Regression Model (#2), using the RandomOverSampler module, predicts a healthy/low-risk loan with 100% precision, while it predicts a high-risk loan with a lower precision at 87%; both precision percentages were the same as in the original logistic regression model. However, the balanced accuracy for the resampled logistic regression model is 99.6%, in comparison to 94.4% for the original logistic regression model.  Similarly, the f-1 and recall scores were higher in the resampled logistic regression model.

Therefore, If the goal of the model is to determine the likelihood of high-risk loans, neither models result in above 90% precision score. However, the Logistic Regression Model (#2) results in fewer false predictions for the testing data and would be the beter model to use based on its high accuracy and recall scores.

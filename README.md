# KWC


## Classifying keywords


In this repository, different models to classify keywords are being developed. 

### Data exploration, manipulation and cleaning

The data consist in a json file that contains keywords associated with different classifications for that keywords. These different classifications are stored in arrays inside the json file. Due to the fact that the categories were stored in arrays, an exploitation of the arrays was done (however, this is something that may be managed in a different way… in base of the accuracy, this may be considered as a hyperparameter).

Once the categories were exploded, the next step was to explore the data in order to adequately clean and normalize the data to have it in a proper structure ready to be used to fit the models. In the process the NaN were looked at, as well as another anomaly that could be detected.

The data has a lot of different languages, and some of them were causing issues in the training, that was why a detection of languages and cleaning was done. Only the keywords in English were kept. 


### Training the model, data preprocessing

In order to have a good model, the categories were counted, and only the top 10 categories were considered. (This could be another hyperparameter that could be changed)

The data was vectorized considering dropping the stop words. After that a TFIDF transformation was applied to the data.

### Models:

Different models were trained with the data preprocessed as described above. All of them gave an accuracy around 46%

Logistic Regression
Naive Bayes Classifier
Support Vector Machines
CNN neural network

#### Logistic Regression:

The logistic regression model gave the more poor accuracy: 46.15%

#### Naive Bayes Classifier

NB gave an accuracy of: 46.26%

#### Support Vector Machines:

Among these classifiers SVM was the one that gave the best accuracy: 47.42%


#### Deep Neural Network: CNN

For the CNN neural network, it was difficult to converge after 5 epochs. Trying with 10 epochs would be the next step, however as the rate of validation is very low, a better accuracy than the accuracy obtained with the conventional ML methods is very unlikely to reach.



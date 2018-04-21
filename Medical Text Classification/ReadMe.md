# Introduction
Medical Text Classification problem has training dataset that contains 14438 records and the test dataset that contains 14442 records. The training dataset consists of the medical abstract   and the class label for that abstract. To find the class label for the medical abstract, first we need to convert the text data into numerical feature vector form which can be used by classifier to predict the output class.

# Text Processing
The first step in the process of building a model is to process the text data. The training and test data is read and stored as a list where each abstract is on different line. The training data set contains class label and medical abstract. The training data is divided by slicing the list after first character of the list of strings. This way training data has now two lists. first list contains all labels and the second list contains all abstracts. scikit-learn provides utilities to extract numerical features from text content. These utilities are:  
tokenizing strings and giving an integer id for each possible token, by splitting the strings using separators like white-spaces and punctuation.
counting the occurrences of tokens in each document.
normalizing and weighting with deciding occurrences as samples / documents.
Here each token is considered a feature for the given dataset. As the document contains many tokens/features and a document only contains few distinct words from all those features, the features which are present is a document are stored in memory. This is called sparse datasets.

Scikit-learn library provides 3 methods to convert text data into sparse datasets. First method is called CountVectorizer. In this 
method, tokenizing and filtering of words like ‘a’, ‘an’, and ‘the’ are performed to convert text data into feature vector. The index of the word is linked with the frequency of occurrence. I used count vectorizer but it was giving low accuracy for the classifier so I opted for different method.
 
I decided to use TFidfVectorizer. In CountVectorizer, occurrence count is different for long and short document. If high, it can affect the model performance. To avoid this, frequency of each word in a document is divided by total no of words in the document which is called term frequency. also, the features with high occurrences is downscaled as they are less informative compared to words which occur less in the dataset. 

I have converted the training dataset and testing dataset lists into panda data frames just for the ease of processing. For the training data frame, I used fit-transform () method of scikit-learn TfidfVectorizer to convert training dataset text data into sparse dataset. Then I used fit () method of TfidfVectorizer to fit the testing dataset into same dimension sparse vector.

# Methodology
## Approach: There are various ways for evaluating any model. I used accuracy and F1-score for evaluating my model. Before applying TfidfVectorizer, I used train_test_split function of scikit-learn library
to split my training data for training and testing purpose. I used 10 % of my training data for testing the model.  

## Model Selection: I applied various models to train my dataset. I started with creating a Naïve Bayes classifier from scikit-learn library. Naïve Bayes classifier was giving very less accuracy and F1score. So, I decided to try another classifier for the dataset. I used logistic regression classifier from scikit-learn library. This classifier gave better accuracy than Naïve bayes classifier but it was giving 0.68 F1 score which I though could be improved. I further tried using one more classifier from scikit-kearn library. I used linear Support Vector Machine with Stochastic Gradient Descent learning. This classifier gave best accuracy compared to other classifiers.  Also, it was giving 0.77 F1 score. To improve F1 score I tried to ensemble multiple machine learning algorithm. 

I designed ensemble classifier using Naïve bayes, Logistic Regression and SVM classifiers. To ensemble classifiers, I used voting ensemble method. Voting is a way of combining predictions from multiple machine learning algorithms.  This ensemble classifier was the slowest classifier amongst the classifier I used for this assignment. And, this classifier was giving less accuracy compared to previously used SVM classifier. Finally, I decided to improve F1 score by improving linear SVM classifier. 
  
For Improving linear SVM with Stochastic Gradient Descent, I changed parameters of SGDClassifier. Some of the parameterts which affected performance of SGDC classfiers are loss, penalty and n_iter. Some of the loss functions provided by scikit-learns are hinge, square_hinge, log, epsilon_insensitive and so on. I tried various combinations of this loss function with penalty(l1,l2 or elasticnet).  Another parameter which affects the performance of SGDC classifier is n_iter which is no of epochs for the training. After running the SGDC classifier several times, I found the SGDC was giving the best F1 score of 0.7968 with loss=’hinge’, penalty=’ elasticnet’  and n_iter= 50.

After finding the this SGDC classifier, I trained this classifier for the entire training set and applied it to test data.

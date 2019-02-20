# nlp_project

This project contains several machine learning models trained from Large Movie Review Dataset.



## Flask web api

This flask web api can predict the sentimental tendency (positive or negative) and highlight sentimental keywords based on your input text. Logistic regression (accuracy 0.877), support vector classifier (accuracy 0.865) and naive-bayes (accuracy: 0.84) are included. If you want to try out please download **nlp_flask** and type in the terminal **python runserver.py**

requirements: keras, nltk (download stopwords), sklearn, pickle, pandas


## Other model

A LSTM RNN model is developed, please refer to the jupyter notebook.

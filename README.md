# Natural-Language-Processing-with-Disaster-Tweets
Natural Language Processing with Disaster Tweets utilizing RNN with LSTM

# Natural Language Processing with Disaster Tweets

## Kaggle Project: 
Link: https://www.kaggle.com/c/nlp-getting-started/overview

## Overview: 
In this competition, the task is to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. The columns for the training and test datasets are:

- `id` a unique identifier for each tweet
- `text` the text of the tweet
- `location` the location the tweet was sent from (may be blank)
- `keyword` a particular keyword from the tweet (may be blank)
- `target` in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

The selected models were a TF-IDF Vectorizer utilizing Logistic Regression and three different Recurrent Neural Networks (RNN) that use Bidirectional Long Short-Term Memory (LSTM) units. The first RNN model is utilized as a baseline model whereas the following two models represent some hyperparameter tuning. 

# First RNN Model Utilizing LSTM
Before the model was operated, the textual data was further preprocessed by tokenization and padding. The breakdown of the first RNN Model Utilizing LSTM is detailed below:

**Input:**
- `vocab_size`: 20,000 most frequent units stored in the vocabulary. The Out Of Vocabulary (OOV) are placed with a special key in the tokenizer. 
- `max_length`: 100 tokens for input that are padded or truncated.

**Architecture:**
- Embedding Layer
    - `embedding_dim` set to 128
- First Bidirectional LSTM Layer
    - 128 LSTM units utilizing bidirectional processing
- Dropout Layer
    - Dropout rate set to 0.3
- Second Bidirectional LSTM Layer
    - 64 LSTM units utilizing bidirectional processing
- Dropout Layer
    - Dropout rate set to 0.3
- Fully Connected Dense Layer
    - 64 neurons utilizing ReLU activation.
- Ouput Layer
    - 1 neuron utilizing sigmoid activation.
 - Optimizer:
    - Adam with default learning rate
- Loss Function:
    - Binary Crossentropy
- Metrics:
    - Accuracy
 
# Second RNN Model Utilizing LSTM
The model architecture for the Second RNN Model Utilizing LSTM is identical to the First RNN Model Utilizing LSTM with the following hyperparameter adjustments: 

- The dropout rate present in both layers was increased from 0.3 to 0.5
- The vocabulary size was reduced from 20000 to 10000
- An L2 Regularization was added (set to 0.02)

Each of these steps were in an effort to try and reduce overfitting. 

# Third RNN Model Utilizing LSTM
The model architecture for the Third RNN Model Utilizing LSTM is identical to the Second RNN Model Utilizing LSTM with the following hyperparameter adjustments: 

- The learning rate was reduced from the default value (0.001) to 0.0001.
- L2 Regularization increased from 0.02 to 0.03.

# Conclusions:
The model metrics for the three RNN models are detailed below (when first ran):

- Model 1 - Accuracy: 0.9304, Precision: 0.9022, Recall: 0.9384, F1-Score: 0.9199
- Model 2 - Accuracy: 0.9173, Precision: 0.9252, Recall: 0.8767, F1-Score: 0.9003
- Model 3 - Accuracy: 0.9074, Precision: 0.9164, Recall: 0.8613, F1-Score: 0.8880

Based on these results the Model 1 had the best accuracy, recall, and F1-score, whereas Model 2 had the best precision and had the second highest metrics in every other category. Based off the plots for the Accuracy vs. Epochs and the Loss vs. Epochs, Model 1 severly overfits the data; therefore, Model 2 is a better overall selection. While Model 3 addresses the overfitting even more, Model 2 seems to provide a good balance of both performance and fit. 

The results from the TF-IDF Vectorization and Logistic Regression model were not as accurate as the RNN models based on the results above. 

Next steps could include further tuning of hyperparameters, such as adding Batch Normalization, tuning the learning rate, L2 Regularization factor, and other features in the layers of the model. 

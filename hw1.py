from re import sub
import seaborn as sb
from sklearn import metrics
from sklearn import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import tarfile
from sklearn.feature_extraction import text as text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing
from pathlib import Path
import string
import re
sb.set_theme(style='darkgrid')


# region Read the data

# A method for extracting data from tsv files
def read_tsv(tar, fname):
    member = tar.getmember(fname)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels

# Update the terminal
print("\nReading the data...")

# Open the tarfile
tar = tarfile.open("data/sentiment.tar.gz", "r:gz")
trainname = "train.tsv"
devname = "dev.tsv"
for member in tar.getmembers():
    if 'train.tsv' in member.name:
        trainname = member.name
    elif 'dev.tsv' in member.name:
        devname = member.name
            
# Create the sentiment object
class Data: pass
sentiment = Data()

# Add the training data
sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)

# Add the testing data
sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)

# Close the tar file
tar.close()

# endregion Read the data

# region 2.1 - TF-IDF and Changing ngram and C-Value

# A method for transforming the data
def transform_data(n):
    """This method transforms the data into a tokenizd form.
    
    :param n: The maximum ngram to encode
    """

    global sentiment
    sentiment.count_vect = TfidfVectorizer(ngram_range=(1,n), stop_words='english')
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)

# Test a range of n values
for n in range(3, 3):

    print("\n--------------------------------------------------------")
    print(f"\nTesting C-values for {n}-grams\n")

    # Transform the data
    transform_data(n)

    # Create lists of test and training accuracies, and c values
    train_accuracy = []
    test_accuracy = []
    c_values = []

    # Test a range of c values
    for c in np.linspace(-4, 4, 9):

        # Print an update
        print(f"Training classifier for C = {str(c).ljust(5, '0')}", end='')

        # Train the classifier at the current c value
        cls = LogisticRegression(C=10**c, random_state=0, solver='lbfgs', max_iter=10000)
        cls.fit(sentiment.trainX, sentiment.trainy)

        # Evaluate the classifier
        train_prediction = cls.predict(sentiment.trainX)
        test_prediction = cls.predict(sentiment.devX)
        train_acc = round(metrics.accuracy_score(sentiment.trainy, train_prediction), 3)
        test_acc = round(metrics.accuracy_score(sentiment.devy, test_prediction), 3)
        print(f" - Train Acc = {str(train_acc).ljust(5, '0')}, " + 
                 f"Test Acc = {str(test_acc).ljust(5, '0')}")

        # Store the accuracies and c value
        train_accuracy += [(train_acc)]
        test_accuracy += [test_acc]
        c_values += [c]

    # Plot the accuries over c
    data = pd.DataFrame({'C Values': c_values, 
                        'Train Accuracy': train_accuracy, 
                        'Test Accuracy': test_accuracy})
    fig, ax = plt.subplots()
    train_line = sb.lineplot(x='C Values', y="Train Accuracy", data=data)
    test_line = sb.lineplot(x='C Values', y="Test Accuracy", data=data)
    ax.set(xlabel='C Values', ylabel="Accuracy", title=f'{n}-gram accuracies over c values')
    ax.legend(labels=["Train Accuracy", "Test Accuracy"])
    plt.show()

    # Save the plot
    fig.savefig(f'plots/hw1-2.1-{n}gram.png')

    # Save the datatable
    data.to_csv(Path(f'tables/{n}gram.csv'), index=False)

# endregion 2.1 - TF-IDF and Changing C-Value

# region 2.2 - Data Cleaning/Refinement

# The new training data
tokenized_train_data = sentiment.train_data.copy()
tokenized_test_data = sentiment.dev_data.copy()


def tokenize(data, vectorizer):
    """This method transforms a list of strings into tokens, in place.

    :param data: An iterable of data
    """

    # Tokenize the entire dataset
    for i in range(len(data)): 

        # Make it all lowercase
        data[i] = data[i].lower()

        # Remove punctuation
        data[i] = ''.join([c if c != '-' else " " for c in data[i]])
        data[i] = ''.join([c for c in data[i] if c not in string.punctuation])

        # Replace all whitespace with a single space
        data[i] = re.sub(r"( |  |\n)+", " ", data[i])

        # Turn the string into a list 
        data[i] = data[i].split()

        # Replace stop words
        data[i] = ''.join([w + " " for w in data[i] if w not in text.ENGLISH_STOP_WORDS])

    return data

# Tokenize the testing and training dataset
vectorizer = TfidfVectorizer(ngram_range=(1,3))
tokenized_train_data = vectorizer.fit_transform(tokenize(tokenized_train_data, vectorizer))
tokenized_test_data = vectorizer.transform(tokenize(tokenized_test_data, vectorizer))

# Vectorize the labels
sentiment.le = preprocessing.LabelEncoder()
sentiment.le.fit(sentiment.train_labels)
sentiment.target_labels = sentiment.le.classes_
sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
sentiment.devy = sentiment.le.transform(sentiment.dev_labels)

# Use the best version of the model
cls = LogisticRegression(C=20, random_state=0, solver='lbfgs', max_iter=10000)

# Train the model
cls.fit(tokenized_train_data, sentiment.trainy)

# Evaluate the classifier
train_prediction = cls.predict(tokenized_train_data)
test_prediction = cls.predict(tokenized_test_data)
train_acc = round(metrics.accuracy_score(sentiment.trainy, train_prediction), 3)
test_acc = round(metrics.accuracy_score(sentiment.devy, test_prediction), 3)
print(f" - Train Acc = {str(train_acc).ljust(5, '0')}, " + 
            f"Test Acc = {str(test_acc).ljust(5, '0')}")

# endregion 2.2 - Data Cleaning/Refinement

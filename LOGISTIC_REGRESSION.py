import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.linear_model import LogisticRegression

m_testDataFile = ".//evaluation_data//test.data"
m_devDataFile = ".//evaluation_data//dev.data"
m_trainDataFile = ".//evaluation_data//train.data"


# Define column names for data
m_columns = ["Topic_Id", "Topic_Name", "Sent_1",
             "Sent_2", "Label", "Sent_1_tag", "Sent_2_tag"]

#################################### HELPER FUNCTIONS #####################################
# Step 1: Load the dataset

#
# Helper to load the dataset
#


def loadDataset(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=m_columns)
    # Strip leading and trailing spaces from all fields
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return data

#
# Ref: https://github.com/cocoxu/SemEval-PIT2015/blob/master/README.md
#


def loadDevDataset(file_path):
    # Read the data file into a DataFrame
    data = loadDataset(file_path)

    # Normalize the data with the following rules
    # paraphrases: (3, 2) (4, 1) (5, 0)
    # non-paraphrases: (1, 4) (0, 5)
    # debatable: (2, 3)  which you may discard if training binary classifier
    data['Label'] = data['Label'].apply(
        lambda x: 1.0 if x.strip() in ["(3, 2)", "(4, 1)", "(5, 0)"] else 0.0)

    return data

#
# The Train and Dev dataset have the same format for the "Label" column
# I re-use the same function within to transform the train data
#


def loadTrainDataset(file_path):
    return loadDevDataset(file_path)


#
# The test dataset has a single numeric entry for the "Label" column
# We need to transform this into a binary Label (0 or 1)
#
def loadTestDataset(file_path):
    data = loadDataset(file_path)
    # Masasage the data with the following rules
    #   The "Label" column for *test data* is in a format of a single digit between
    #   between 0 (no relation) and 5 (semantic equivalence), annotated by expert.
    #   We would suggest map them to binary labels as follows:

    #     paraphrases: 4 or 5
    #     non-paraphrases: 0 or 1 or 2
    #     debatable: 3   which we discarded in Paraphrase Identification evaluation
    data['Label'] = data['Label'].apply(
        lambda x: 1.0 if (x > 3) else 0.0)

    return data


def compute_cosine_similarity(sentences_1, sentences_2):
    # Vectorize the sentences. Use 'fit' on the entire dataset to ensure vocabulary consistency between comparisons
    vectorizer = TfidfVectorizer()
    all_sentences = sentences_1 + sentences_2
    vectorizer.fit(all_sentences)

    vectors_1 = vectorizer.transform(sentences_1)
    vectors_2 = vectorizer.transform(sentences_2)

    # Compute cosine similarity between pairs of sentences
    cosine_similarities = [cosine_similarity(
        v1, v2)[0][0] for v1, v2 in zip(vectors_1, vectors_2)]
    # Reshape for compatibility with sklearn models
    return np.array(cosine_similarities).reshape(-1, 1)


def preprocess(df):
    cosine_similarities = compute_cosine_similarity(
        df['Sent_1'], df['Sent_2'])
    labels = df['Label'].values
    return cosine_similarities, labels


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("F1 Score:", f1_score(y_test, predictions))
    print("Accuracy Score:", accuracy_score(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))


def fine_tune_model(X_train, y_train):

    parameters = {'C': [0.1, 1, 10, 100], 'solver': [
        'liblinear', 'newton-cg', 'lbfgs']}
    clf = GridSearchCV(LogisticRegression(max_iter=10000),
                       parameters, scoring='f1_macro')
    clf.fit(X_train, y_train)

    print("Best Hyper Parameters:", clf.best_params_)
    return clf.best_estimator_


if __name__ == "__main__":

    devDataset = loadDevDataset(m_devDataFile)
    trainDataset = loadTrainDataset(m_trainDataFile)
    testDataset = loadTestDataset(m_testDataFile)

    sentences_test, labels_test = preprocess(testDataset)
    sentences_train, labels_train = preprocess(trainDataset)

    X_train = sentences_train    # the cosine similarity
    y_train = labels_train

    X_test = sentences_test
    y_test = labels_test

    # X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=10000)

    print("----- LOGISTIC REGRESSION Training Model -----")
    model.fit(X_train, y_train)

    print("-----  LOGISTIC REGRESSION Model Evaluation -----")
    evaluate_model(model, X_test, y_test)

    print("----- LOGISTIC REGRESSION Fine-tuning Model -----")
    tuned_model = fine_tune_model(X_train, y_train)
    print("----- LOGISTIC REGRESSION Tuned Model Evaluation -----")
    evaluate_model(tuned_model, X_test, y_test)

    # Save the fine-tuned model
    joblib.dump(tuned_model, 'logistic_regression_paraphrase_model.pkl')

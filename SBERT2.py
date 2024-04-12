

from sklearn.metrics import f1_score, accuracy_score, classification_report

import numpy as np
from sentence_transformers import util
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import torch
from scipy.optimize import minimize_scalar


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
        lambda x: 1.0 if (x >= 3) else 0.0)

    return data


# Define a simple accuracy metric

def accuracy(out_features, labels):
    cos_scores = util.pytorch_cos_sim(out_features[0], out_features[1])
    predictions = np.argmax(cos_scores.cpu().numpy(), axis=1)
    acc = np.sum(predictions == labels) / len(labels)
    return acc


# def evaluate_sbert_model(model, test_df):
#     # Generate embeddings
#     embeddings1 = model.encode(
#         test_df['Sent_1'].tolist(), convert_to_tensor=True)
#     embeddings2 = model.encode(
#         test_df['Sent_2'].tolist(), convert_to_tensor=True)

#     # Convert embeddings to numpy arrays (for compatibility) and save using pickle
#     with open('SBERT_embeddings.pkl', 'wb') as f:
#         pickle.dump({
#             "embeddings1": embeddings1.cpu().numpy(),
#             "embeddings2": embeddings2.cpu().numpy()
#         }, f)

#     # Compute cosine similarities
#     cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

#     # Assuming you want to compare each item in embeddings1 with the corresponding item in embeddings2
#     # Extract diagonal elements representing the similarity scores for corresponding sentence pairs
#     similarity_scores_for_pairs = torch.diag(cosine_scores).cpu()

#     # Define a threshold for deciding whether a pair is considered a paraphrase
#     threshold = 0.5
#     predictions = (similarity_scores_for_pairs > threshold).int()

#     # Compute metrics
#     # Ensure to convert predictions tensor to a list or numpy array for scikit-learn compatibility
#     print("Accuracy Score:", accuracy_score(
#         test_df['Label'].tolist(), predictions.cpu().numpy()))
#     print("F1 Score:", f1_score(
#         test_df['Label'].tolist(), predictions.cpu().numpy()))
#     print("Classification Report:\n", classification_report(
#         test_df['Label'].tolist(), predictions.cpu().numpy()))

def optimize_threshold(y_true, scores):
    """
    Optimize the decision threshold based on F1 score.
    """
    def f1_neg(threshold):
        predictions = (scores > threshold).astype(int)
        return -f1_score(y_true, predictions)  # Negative F1 for minimization

    result = minimize_scalar(f1_neg, bounds=(0, 1), method='bounded')
    return result.x


def evaluate_sbert_model_improved(model, test_df):
    """
    Generate embeddings, compute cosine similarities, optimize the decision threshold,
    and compute metrics.
    """
    # Generate embeddings
    embeddings1 = model.encode(
        test_df['Sent_1'].tolist(), convert_to_tensor=True).cpu().numpy()
    embeddings2 = model.encode(
        test_df['Sent_2'].tolist(), convert_to_tensor=True).cpu().numpy()

    # Save embeddings
    with open('SBERT_embeddings_improved.pkl', 'wb') as f:
        pickle.dump({"embeddings1": embeddings1,
                    "embeddings2": embeddings2}, f)

    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(
        torch.tensor(embeddings1), torch.tensor(embeddings2))

    # Check if cosine_scores is square and extract diagonal if it is
    if cosine_scores.size(0) == cosine_scores.size(1):
        similarity_scores_for_pairs = torch.diag(cosine_scores).cpu().numpy()
    else:
        raise ValueError(
            "The number of samples in embeddings1 and embeddings2 do not match.")

    # Optimize the decision threshold
    y_true = test_df['Label'].tolist()
    optimized_threshold = optimize_threshold(
        y_true, similarity_scores_for_pairs)

    # Generate predictions based on optimized threshold
    predictions = (similarity_scores_for_pairs >
                   optimized_threshold).astype(int)

    # Compute metrics
    print(f"Optimized Threshold: {optimized_threshold}")
    print("Accuracy Score:", accuracy_score(y_true, predictions))
    print("F1 Score:", f1_score(y_true, predictions))
    print("Classification Report:\n", classification_report(y_true, predictions))


if __name__ == "__main__":

    devDataset = loadDevDataset(m_devDataFile)
    trainDataset = loadTrainDataset(m_trainDataFile)
    testDataset = loadTestDataset(m_testDataFile)

    model = SentenceTransformer(
        'sentence-transformers_paraphrase_model')

    # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # # Convert the DataFrame into a list of InputExample objects
    # train_examples = [InputExample(texts=[row['Sent_1'], row['Sent_2']],
    #                                label=row['Label']) for index, row in trainDataset.iterrows()]
    # train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # # Use the MultipleNegativesRankingLoss for the paraphrase identification task
    # train_loss = losses.MultipleNegativesRankingLoss(model)

    # # Training SBERT with fine-tuning
    # model.fit(train_objectives=[(train_dataloader, train_loss)],
    #           epochs=4,
    #           warmup_steps=100,
    #           show_progress_bar=True,
    #           evaluator=None,  # You can define an evaluator for validation during training
    #           output_path='sentence-transformers_paraphrase_model')  # Save the model

    evaluate_sbert_model_improved(model, testDataset)

    # Saving was done during training with output_path='sentence-transformers_paraphrase_model'
    # To load:
    # loaded_model = SentenceTransformer(
    #     'sentence-transformers_paraphrase_model')

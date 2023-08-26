import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

train_data = pd.read_csv("train_stop_clean_lemma.csv")
test_data = pd.read_csv("test_stop_clean_lemma.csv")

# print(train_data)

# # Remove nan col;
# if "Unnamed: 2" in list(train_data.columns):
#   temp = train_data[train_data["Unnamed: 2"].isnull()]
#   train_data = train_data.iloc[list(temp.index), :][["label", "text"]]
# train_data = train_data[train_data["text"].notnull()]
# train_data = train_data[train_data["label"].notnull()]
# train_data = train_data.reset_index(drop = True)
# if "Unnamed: 2" in list(test_data.columns):
#   temp = test_data[test_data["Unnamed: 2"].isnull()]
#   test_data = test_data.iloc[list(temp.index), :][["label", "text"]]
# test_data = test_data[test_data["text"].notnull()]
# test_data = test_data[test_data["label"].notnull()]
# test_data = test_data.reset_index(drop = True)

corpus = pd.concat((train_data["text"], test_data["text"]))
tfidf_model = TfidfVectorizer().fit(corpus)

tfidf_train = tfidf_model.transform(train_data["text"])
tfidf_test = tfidf_model.transform(test_data["text"])

# for i in range(train_data.shape[0]):
#     if type(train_data["text"][i]) == str:
#         train_data["label"][i] = int(train_data["label"][i])

# for i in range(test_data.shape[0]):
#     if type(test_data["text"][i]) == str:
#         test_data["label"][i] = int(test_data["label"][i])

# y_train = train_data["label"] - 1
# y_test = test_data["label"] - 1

y_train = train_data["label"].astype("int")
y_test = test_data["label"].astype("int")
y_train = y_train - 1
y_test = y_test - 1

# print(tfidf_train.shape)
# print(y_train.shape)

# print(y_train)
# print(set(y_train))

lr = LogisticRegression()
lr.fit(tfidf_train, y_train)
lr_test_pred = lr.predict(tfidf_test)

acc = accuracy_score(lr_test_pred, y_test)
prec = precision_score(lr_test_pred, y_test, average = "macro")
rec = recall_score(lr_test_pred, y_test, average = "macro")
f1 = f1_score(lr_test_pred, y_test, average = "macro")
print("Accuracy: {}".format(acc))
print("Precision: {}".format(prec))
print("Recall: {}".format(rec))
print("F1: {}".format(f1))

test_data["pred"] = lr_test_pred
test_data.to_csv("LR_pred.csv")

# clf = MultinomialNB()
# clf.fit(tfidf_train, df_train["Label"])
# pred_results = clf.predict(tfidf_test)

# results = df_test.drop("Text", axis = "columns")
# results["Predicted"] = pred_results
# results.to_csv("Pred.csv", index = False)

print(confusion_matrix(lr_test_pred, y_test))
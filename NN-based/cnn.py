import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data as Data
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchtext.vocab import GloVe
glove = GloVe(name='6B', dim=200)
import nltk
nltk.download('punkt')
from nltk import word_tokenize
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

train_data = pd.read_csv("train_stop_clean_lemma.csv")
test_data = pd.read_csv("test_stop_clean_lemma.csv")
valid_data = pd.read_csv("val_stop_clean_lemma.csv")

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

print("The number of train data: {}; The number of test data: {}".format(train_data.shape[0], test_data.shape[0]))

max_len = 128
gpu = True

# vecs_lst_test = []
# for i in range(test_data.shape[0]): #test_data.shape[0]
#   # print(i)
#   if type(test_data["label"][i]) == str:
#     test_data["label"][i] = int(test_data["label"][i])
#   w_lst = word_tokenize(test_data["text"][i])
#   if len(w_lst) >= max_len:
#     w_lst = w_lst[:max_len]
#   else:
#     for t in range(max_len - len(w_lst)):
#       w_lst.append("unk")
#   s_vecs_test = []
#   for w in w_lst:
#     s_vecs_test.append(glove.get_vecs_by_tokens(w).unsqueeze(0))
#   s_vec_test = torch.cat(s_vecs_test, 0)
#   vecs_lst_test.append(s_vec_test.unsqueeze(0))
# # print(c.shape)
# X_test = torch.cat(vecs_lst_test, 0)
# # print(X_test.shape)
# y_test = torch.tensor(test_data["label"]) - 1
# # print(y_test.shape)

# torch.save(X_test, "x_test.pt")
# torch.save(y_test, "y_test.pt")

# vecs_lst_train = []
# for i in range(train_data.shape[0]): #train_data.shape[0]
#   # print(i)
#   if type(train_data["label"][i]) == str:
#     train_data["label"][i] = int(train_data["label"][i])
#   w_lst = word_tokenize(train_data["text"][i])
#   if len(w_lst) >= max_len:
#     w_lst = w_lst[:max_len]
#   else:
#     for t in range(max_len - len(w_lst)):
#       w_lst.append("unk")
#   s_vecs_train = []
#   for w in w_lst:
#     s_vecs_train.append(glove.get_vecs_by_tokens(w).unsqueeze(0))
#   s_vec_train = torch.cat(s_vecs_train, 0)
#   vecs_lst_train.append(s_vec_train.unsqueeze(0))
# # print(torch.cat(vecs_lst_test, 0).shape)
# X_train = torch.cat(vecs_lst_train, 0)
# y_train = torch.tensor(train_data["label"]) - 1

# torch.save(X_train, "x_train.pt")
# torch.save(y_train, "y_train.pt")

# vecs_lst_valid = []
# for i in range(valid_data.shape[0]): #test_data.shape[0]
#   # print(i)
#   if type(valid_data["label"][i]) == str:
#     valid_data["label"][i] = int(valid_data["label"][i])
#   w_lst = word_tokenize(valid_data["text"][i])
#   if len(w_lst) >= max_len:
#     w_lst = w_lst[:max_len]
#   else:
#     for t in range(max_len - len(w_lst)):
#       w_lst.append("unk")
#   s_vecs_valid = []
#   for w in w_lst:
#     s_vecs_valid.append(glove.get_vecs_by_tokens(w).unsqueeze(0))
#   s_vec_valid = torch.cat(s_vecs_valid, 0)
#   vecs_lst_valid.append(s_vec_valid.unsqueeze(0))
# # print(c.shape)
# X_valid = torch.cat(vecs_lst_valid, 0)
# # print(X_test.shape)
# y_valid = torch.tensor(valid_data["label"]) - 1
# # print(y_test.shape)

# torch.save(X_valid, "x_valid.pt")
# torch.save(y_valid, "y_valid.pt")

X_train = torch.load("x_train.pt")
y_train = torch.load("y_train.pt")
X_test = torch.load("x_test.pt")
y_test = torch.load("y_test.pt")
X_valid = torch.load("x_valid.pt")
y_valid = torch.load("y_valid.pt")

# torch.save(X_train, 'X_train.pt')
# torch.save(y_train, 'y_train.pt')
# torch.save(X_test, 'X_test.pt')
# torch.save(y_test, 'X_test.pt')

# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15)

def print_grad(grad):
    print(grad)

# class LSTM_Net(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, num_layers, dropout=0.5):
#         super(LSTM_Net, self).__init__()
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
#         self.dropout = nn.Dropout(dropout)
#         self.Linear1 = nn.Linear(hidden_dim, 32)
#         self.Linear2 = nn.Linear(32, 4)
#         self.softmax = nn.LogSoftmax(dim = 1)
#         self.act = nn.ReLU()

#     def forward(self, inputs):
#         x, _ = self.lstm(inputs, None)
#         x = x[:, -1, :] 
#         # x = self.dropout(x)
#         # x = self.act(x)
#         x = self.Linear1(x)
#         # x = self.dropout(x)
#         # x = self.act(x)
#         x = self.Linear2(x)
#         x = self.softmax(x)
#         return x

# class LSTMClassifier(nn.Module):

#     def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes):
#         super(LSTMClassifier, self).__init__()
#         # self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, num_classes)
#         self.tanh = nn.Tanh()
#         self.dropout = nn.Dropout(0.3)
#         self.w = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)

#     def forward(self, x):
#         # x = self.embedding(x)
#         out, (hidden, _) = self.lstm(x)
#         m = self.tanh(out)
#         score = torch.matmul(m, self.w)
#         alpha = F.softmax(score, dim=0).unsqueeze(-1)
#         output_attention = out * alpha
#         hidden = torch.reshape(hidden, [out.shape[0], -1, out.shape[2]])
#         hidden = torch.mean(hidden, dim=1)
#         output_attention = torch.sum(output_attention, dim=1)
#         out = torch.sum(out, dim=1)
#         fc_input = self.dropout(out + output_attention + hidden)
#         out = self.fc(fc_input)
#         return out

class TextCNN(nn.Module):

    def __init__(self, embedding_dim, num_classes, kernel_sizes=[3, 4, 5], num_filters=100):
        super(TextCNN, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]

        # Apply convolutions and max-pooling
        conv_outputs = []
        for conv in self.convs:
            conv_output = conv(x)  # [batch_size, num_filters, seq_len - kernel_size + 1, 1]
            activation = F.relu(conv_output)
            max_pooled = F.max_pool2d(activation, (conv_output.shape[2], 1))  # [batch_size, num_filters, 1, 1]
            conv_outputs.append(max_pooled.squeeze(-1).squeeze(-1))  # [batch_size, num_filters]

        # Concatenate and apply dropout
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)

        # Apply linear layer
        x = self.fc(x)
        return x

BATCH_SIZE = 512

y_train = torch.tensor(np.array(y_train).astype(float))
torch_trainset = Data.TensorDataset(X_train, y_train)

train_loader = Data.DataLoader(
dataset=torch_trainset,    
batch_size=BATCH_SIZE,     
shuffle=True,          
num_workers=2,              
drop_last = True,
)

y_test = torch.tensor(np.array(y_test).astype(float))
torch_testset = Data.TensorDataset(X_test, y_test)

test_loader = Data.DataLoader(
dataset=torch_testset,    
batch_size=BATCH_SIZE,     
shuffle=True,          
num_workers=2,              
drop_last = True,
)

y_valid = torch.tensor(np.array(y_valid).astype(float))
torch_validset = Data.TensorDataset(X_valid, y_valid)

valid_loader = Data.DataLoader(
dataset=torch_validset,    
batch_size=BATCH_SIZE,     
shuffle=True,          
num_workers=2,              
drop_last = True,
)

################################################################
#Training
################################################################

# model = LSTM_Net(200, 16, 12, 0.2)
# model = LSTMClassifier(200, 128, 2, 4)
model = TextCNN(200, 4, [3, 4, 5], 100)

num_epoch = 32
batch_size = BATCH_SIZE
num_labels = 4 # Must be consistent with the input data's size
learning_rate = 0.01
weight_decay = 0.0001
learning_rate_decay = 0.9
min_loss = 10000000
train_loss = []
valid_loss = []
train_acc = []
lr_list = []
# train_recall = []

if torch.cuda.is_available() and gpu:
  print("Run on GPU!")
else:
  print("Run on CPU!")

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch): # num_epoch

  if (epoch + 1) % 3 == 0:
    for p in opt.param_groups:
      p['lr'] *= learning_rate_decay
  
  lr_list.append(opt.state_dict()['param_groups'][0]['lr'])
  
  for step, (X_train, y_train) in enumerate(train_loader):

    model.train()

    if torch.cuda.is_available() and gpu:
      X_train = X_train.to("cuda")
      y_train = y_train.to("cuda")
      model = model.to("cuda")
    outputs = model(X_train)

    # model.fc.weight.register_hook(print_grad)

    loss = loss_fn(outputs, y_train.long())
    opt.zero_grad()
    loss.backward()
    opt.step()

    # print(outputs)

    outputs_labels = torch.max(outputs, axis = 1)[1]
    num_true_pred = (outputs_labels == y_train.long()).sum().item()
    accuracy = num_true_pred / y_train.long().nelement()

    # ner_labels = train_labels != 0
    # true_ners = (outputs_labels == train_labels) & ner_labels
    # num_ners = ner_labels.sum().item()
    # recall = true_ners.sum().item() / num_ners

    # optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=0.01, eps=0.000001)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    print("Epoch: {}, Step: {}, Loss: {}, Accuracy:{}".format(epoch, step, loss, accuracy))

    train_loss.append(loss.item())
    train_acc.append(accuracy)
    # train_recall.append(recall)
    

    model.eval()
    for eval_step, (X_valid, y_valid) in enumerate(valid_loader):

      if torch.cuda.is_available() and gpu:
          X_valid = X_valid.to('cuda')
          y_valid = y_valid.to('cuda')
          
      eval_loss = 0
      eval_outputs = model(X_valid)
      temp_eval_loss = loss_fn(eval_outputs, y_valid.long())
      eval_loss += temp_eval_loss

      if eval_loss < min_loss:
          print("original loss: {}, new loss: {}".format(min_loss, eval_loss))
          min_loss = eval_loss
          torch.save(model.state_dict(), "model_cnn.pth")
          print("save")
      valid_loss.append(eval_loss.item())

plt.figure(figsize=(10, 6))
plt.plot(train_loss)
plt.xlabel("batches")
plt.ylabel("train_loss")
plt.title("train loss v.s. epochs")
plt.savefig("train_loss.jpg")

plt.figure(figsize=(10, 6))
plt.plot(train_acc)
plt.xlabel("batches")
plt.ylabel("train_accuracy")
plt.title("train accuracy v.s. epochs")
plt.savefig("train_accuracy.jpg")

# plt.figure(figsize=(10, 6))
# plt.plot(train_recall)
# plt.xlabel("epochs")
# plt.ylabel("train_recall")
# plt.title("train recall v.s. epochs")
# plt.savefig("train_recall.jpg")

plt.figure(figsize=(10, 6))
plt.plot(valid_loss)
plt.xlabel("batches")
plt.ylabel("valid loss")
plt.title("valid loss v.s. epochs")
plt.savefig("valid_loss.jpg")

plt.figure(figsize=(10, 6))
plt.plot(lr_list)
plt.xlabel("batches")
plt.ylabel("learning_rate")
plt.title("learning_rate v.s. epochs")
plt.savefig("learning_rate.jpg")

# # Calculate Metrics
model.eval()
test_loss = 0
test_acc = 0
test_recall = 0
num_true_pred = 0
num_true_ner = 0
test_num = 0
ner_num = 0
for test_step, (X_test, y_test) in enumerate(test_loader):

  if torch.cuda.is_available() and gpu:
    X_test = X_test.to('cuda')
    y_test = y_test.to('cuda')

  test_outputs = model(X_test)
  temp_test_loss = loss_fn(test_outputs, y_test.long())
  test_loss += temp_test_loss

  test_outputs_labels = torch.max(test_outputs, axis=1)[1]
  temp_num_true_pred = (test_outputs_labels == y_test.long()).sum().item()
  test_num += y_test.long().nelement()
  num_true_pred += temp_num_true_pred

  # ner_labels = test_labels != 0
  # true_ners = (test_outputs_labels == test_labels) & ner_labels
  # num_true_ner += true_ners.sum().item()
  # ner_num += ner_labels.sum().item()

test_acc = num_true_pred / test_num
# test_recall = num_true_ner / ner_num

print("Test Results: \nLoss: {}\nAccuracy: {}".format(test_loss, test_acc))
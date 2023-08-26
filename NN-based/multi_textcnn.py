import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchtext.vocab import GloVe
glove = GloVe(name='6B', dim=200)
import nltk
nltk.download('punkt')
from nltk import word_tokenize
import warnings
warnings.filterwarnings("ignore")

from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('stanford-corenlp-4.5.3')

# train_data = pd.read_csv("train_data_pos_parser.csv")
# test_data = pd.read_csv("test_data_pos_parser.csv")
# valid_data = pd.read_csv("valid_data_pos_parser.csv")

# train_data = pd.read_csv("train_data_pos_parser_explore.csv").iloc[:300, :]
# test_data = pd.read_csv("train_data_pos_parser_explore.csv").iloc[:50, :]
# valid_data = pd.read_csv("train_data_pos_parser_explore.csv").iloc[:50, :]

train_data = pd.read_csv("train_masked_stop_lemma.csv")
test_data = pd.read_csv("test_masked_stop_lemma.csv")
valid_data = pd.read_csv("val_masked_stop_lemma.csv")

max_len = 512
gpu = False

# print("Start Embedding!!!")

# vecs_lst_test = []
# for i in range(test_data.shape[0]): #test_data.shape[0]
#   # print(i)
#   if type(test_data["label"][i]) == str:
#     test_data["label"][i] = int(test_data["label"][i])
#   w_lst = word_tokenize(test_data["masked_summary"][i])
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
#   w_lst = word_tokenize(train_data["masked_summary"][i])
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
#   w_lst = word_tokenize(valid_data["masked_summary"][i])
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

X_train = torch.load("x_train_masked_sum.pt")
y_train = torch.load("y_train_masked_sum.pt")
X_test = torch.load("x_test_masked_sum.pt")
y_test = torch.load("y_test_masked_sum.pt")
X_valid = torch.load("x_valid_masked_sum.pt")
y_valid = torch.load("y_valid_masked_sum.pt")

# train_data = train_data.iloc[:300, :]
# test_data = test_data.iloc[:50, :]
# valid_data = valid_data.iloc[:50, :]

# max_len = 128
# gpu = True

train_data["pos_tagger"]  = ""
train_data["parser"] = ""
test_data["pos_tagger"]  = ""
test_data["parser"] = ""
valid_data["pos_tagger"]  = ""
valid_data["parser"] = ""
train_data["pos_tagger_id"]  = ""
train_data["parser_id"] = ""
test_data["pos_tagger_id"]  = ""
test_data["parser_id"] = ""
valid_data["pos_tagger_id"]  = ""
valid_data["parser_id"] = ""


print("Start getting parser and pos tag!!!")
for i in range(train_data.shape[0]): # train_data.shape[0]
    train_pos_str = ""
    train_parse_str = ""
    if i%1000 == 0:
      print(i)
      train_data.to_csv("train_data_pos_parser_explore.csv")
      train_data = pd.read_csv("train_data_pos_parser_explore.csv")
    temp = train_data["masked_summary"][i]
    w_lst = word_tokenize(temp)
    pos_tags =nltk.pos_tag(w_lst)
    pos_lst = []
    dep_lst = []
    for p in pos_tags:
        train_pos_str += p[1]
        train_pos_str += " "
        #pos_lst.append(p[1])
    parsers = nlp.dependency_parse(temp)
    for dep in parsers:
        # print(dep)
        #dep_lst.append(dep[0])
        train_parse_str += dep[0]
        train_parse_str += " "
    train_data["pos_tagger"][i] = train_pos_str.strip(" ")
    train_data["parser"][i] = train_parse_str.strip(" ")
# print(train_data)
train_data.to_csv("train_data_pos_parser_explore.csv")
# print(train_data)

for i in range(test_data.shape[0]): # train_data.shape[0]
    if i%100 == 0:
      print(i)
      test_data.to_csv("test_data_pos_parser_explore.csv")
      test_data = pd.read_csv("test_data_pos_parser_explore.csv")
    test_pos_str = ""
    test_parse_str = ""
    temp = test_data["masked_summary"][i]
    w_lst = word_tokenize(temp)
    pos_tags =nltk.pos_tag(w_lst)
    pos_lst = []
    dep_lst = []
    for p in pos_tags:
        #pos_lst.append(p[1])
        test_pos_str += p[1]
        test_pos_str += " "
    parsers = nlp.dependency_parse(temp)
    for dep in parsers:
        # print(dep)
        #dep_lst.append(dep[0])
        test_parse_str += dep[0]
        test_parse_str += " "
    test_data["pos_tagger"][i] = test_pos_str.strip(" ")
    test_data["parser"][i] = test_parse_str.strip(" ")
# print(train_data)
test_data.to_csv("test_data_pos_parser_exlpore.csv")

for i in range(valid_data.shape[0]): # train_data.shape[0]
    if i%100 == 0:
      print(i)
      valid_data.to_csv("valid_data_pos_parser_explore.csv")
      valid_data = pd.read_csv("valid_data_pos_parser_explore.csv")
    valid_pos_str = ""
    valid_parse_str = ""
    temp = valid_data["masked_summary"][i]
    w_lst = word_tokenize(temp)
    pos_tags =nltk.pos_tag(w_lst)
    pos_lst = []
    dep_lst = []
    for p in pos_tags:
        # pos_lst.append(p[1])
        valid_pos_str += p[1]
        valid_pos_str += " "
    parsers = nlp.dependency_parse(temp)
    for dep in parsers:
        # print(dep)
        # dep_lst.append(dep[0])
        valid_parse_str += dep[0]
        valid_parse_str += " "
    valid_data["pos_tagger"][i] = valid_pos_str.strip(" ")
    valid_data["parser"][i] = valid_parse_str.strip(" ")
valid_data.to_csv("valid_data_pos_parser_explore.csv")

# train_data["pos_tagger_id"]  = ""
# train_data["parser_id"] = ""
# test_data["pos_tagger_id"]  = ""
# test_data["parser_id"] = ""
# valid_data["pos_tagger_id"]  = ""
# valid_data["parser_id"] = ""

# print(valid_data)

# temp = train_data["pos_tagger"][0]
# print(temp.split(", "))

temp = []
for i in range(train_data.shape[0]):
    temp += train_data["pos_tagger"][i].split(" ")
    temp = list(set(temp))
for i in range(test_data.shape[0]):
    temp += test_data["pos_tagger"][i].split(" ")
    temp = list(set(temp))
for i in range(valid_data.shape[0]):
    temp += valid_data["pos_tagger"][i].split(" ")
    temp = list(set(temp))
    
index2pos = dict(enumerate(temp))
index2pos[len(index2pos)] = "padding"
pos2index = {value: key for key, value in index2pos.items()}
    
temp = []
for i in range(train_data.shape[0]):
    temp += train_data["parser"][i].split(" ")
    temp = list(set(temp))
for i in range(test_data.shape[0]):
    temp += test_data["parser"][i].split(" ")
    temp = list(set(temp))
for i in range(valid_data.shape[0]):
    temp += valid_data["parser"][i].split(" ")
    temp = list(set(temp))

index2par = dict(enumerate(temp))
index2par[len(index2par)] = "padding"
par2index = {value: key for key, value in index2par.items()}
print(par2index)

# print(par2index)

# For train dataset
temp_train_pos_tagger_id_tensor = []
temp_train_parser_id_tensor = []


print("Start embedding Parser and Pos Tag!!!")
for i in range(train_data.shape[0]):
    
    # print(i)
    
    temp_train_pos_tagger = train_data["pos_tagger"][i]
    temp_train_parser = train_data["parser"][i]
    # print(temp_train_parser)
    
    temp_train_pos_tagger_id = []
    temp_train_parser_id = []
    temp_train_pos_tagger_id_str = ""
    temp_train_parser_id_str = ""
    
    # print(temp_pos_tagger)
    # print(temp_parser)
    
    temp_train_pos_tagger_lst = temp_train_pos_tagger.split(" ")
    temp_train_parser_lst = temp_train_parser.split(" ")
    
    # print(temp_train_parser_lst)
    
    if len(temp_train_pos_tagger_lst) <= 512:
        for pt in temp_train_pos_tagger_lst:
            
            temp_train_pos_tagger_id.append(pos2index[pt])
            
            temp_train_pos_tagger_id_str += pt
            temp_train_pos_tagger_id_str += " "
            
        for pt_index in range(512 - len(temp_train_pos_tagger_lst)):
            
            temp_train_pos_tagger_id.append(pos2index["padding"])
            
            temp_train_pos_tagger_id_str += "padding"
            temp_train_pos_tagger_id_str += " "
            
        train_data["pos_tagger_id"][i] = temp_train_pos_tagger_id_str
            
    else:
        for pt_index in range(512):
            
            temp_train_pos_tagger_id.append(pos2index[temp_train_pos_tagger_lst[pt_index]])
            
            temp_train_pos_tagger_id_str += temp_train_pos_tagger_lst[pt_index]
            temp_train_pos_tagger_id_str += " "
            
        train_data["pos_tagger_id"][i] = temp_train_pos_tagger_id_str
        
    if len(temp_train_parser_lst) <= 512:
        # print("ok")
        for par in temp_train_parser_lst:
            
            # print(par)
            
            temp_train_parser_id.append(par2index[par])
            
            temp_train_parser_id_str += pt
            temp_train_parser_id_str += " "
            
        for pt_index in range(512 - len(temp_train_parser_lst)):
            
            temp_train_parser_id.append(par2index["padding"])
            
            temp_train_parser_id_str += "padding"
            temp_train_parser_id_str += " "
            
        train_data["parser_id"][i] = temp_train_parser_id_str
    
    else:
        for par_index in range(512):
            
            temp_train_parser_id.append(par2index[temp_train_parser_lst[pt_index]])
            
            temp_train_parser_id_str += temp_train_parser_lst[pt_index]
            temp_train_parser_id_str += " "
            
        train_data["pos_tagger_id"][i] = temp_train_parser_id_str
    
    temp_train_pos_tagger_id_tensor.append(torch.tensor(temp_train_pos_tagger_id).unsqueeze(0))
    temp_train_parser_id_tensor.append(torch.tensor(temp_train_parser_id).unsqueeze(0))
        
    # print(temp_pos_tagger_id)

train_pos_tagger_id = torch.concat(temp_train_pos_tagger_id_tensor)
train_parser_id = torch.concat(temp_train_parser_id_tensor)
torch.save(train_pos_tagger_id, "train_pos_tagger_id.pt")
torch.save(train_parser_id, "train_parser_id.pt")
# train_data.to_csv("train_data_pos_parser_explore.csv")

# print(train_pos_tagger_id.shape)
# print(temp_train_pos_tagger_id_tensor[0].shape)
# print(len(temp_train_pos_tagger_id_tensor))
# print(temp_train_pos_tagger_id_tensor)


# For test dataset 
temp_test_pos_tagger_id_tensor = []
temp_test_parser_id_tensor = []


for i in range(test_data.shape[0]):
    
    # print(i)
    
    temp_test_pos_tagger = test_data["pos_tagger"][i]
    temp_test_parser = test_data["parser"][i]
    # print(temp_train_parser)
    
    temp_test_pos_tagger_id = []
    temp_test_parser_id = []
    temp_test_pos_tagger_id_str = ""
    temp_test_parser_id_str = ""
    
    # print(temp_pos_tagger)
    # print(temp_parser)
    
    temp_test_pos_tagger_lst = temp_test_pos_tagger.split(" ")
    temp_test_parser_lst = temp_test_parser.split(" ")
    
    # print(temp_train_parser_lst)
    
    if len(temp_test_pos_tagger_lst) <= 512:
        for pt in temp_test_pos_tagger_lst:
            
            temp_test_pos_tagger_id.append(pos2index[pt])
            
            temp_test_pos_tagger_id_str += pt
            temp_test_pos_tagger_id_str += " "
            
        for pt_index in range(512 - len(temp_test_pos_tagger_lst)):
            
            temp_test_pos_tagger_id.append(pos2index["padding"])
            
            temp_test_pos_tagger_id_str += "padding"
            temp_test_pos_tagger_id_str += " "
            
        test_data["pos_tagger_id"][i] = temp_test_pos_tagger_id_str
            
    else:
        for pt_index in range(512):
            
            temp_test_pos_tagger_id.append(pos2index[temp_test_pos_tagger_lst[pt_index]])
            
            temp_test_pos_tagger_id_str += temp_test_pos_tagger_lst[pt_index]
            temp_test_pos_tagger_id_str += " "
            
        test_data["pos_tagger_id"][i] = temp_test_pos_tagger_id_str
        
    if len(temp_test_parser_lst) <= 512:
        # print("ok")
        for par in temp_test_parser_lst:
            
            # print(par)
            
            temp_test_parser_id.append(par2index[par])
            
            temp_test_parser_id_str += pt
            temp_test_parser_id_str += " "
            
        for pt_index in range(512 - len(temp_test_parser_lst)):
            
            temp_test_parser_id.append(par2index["padding"])
            
            temp_test_parser_id_str += "padding"
            temp_test_parser_id_str += " "
            
        test_data["parser_id"][i] = temp_test_parser_id_str
    
    else:
        for par_index in range(512):
            
            temp_test_parser_id.append(par2index[temp_test_parser_lst[pt_index]])
            
            temp_test_parser_id_str += temp_test_parser_lst[pt_index]
            temp_test_parser_id_str += " "
            
        train_data["pos_tagger_id"][i] = temp_test_parser_id_str
    
    temp_test_pos_tagger_id_tensor.append(torch.tensor(temp_test_pos_tagger_id).unsqueeze(0))
    temp_test_parser_id_tensor.append(torch.tensor(temp_test_parser_id).unsqueeze(0))
        
    # print(temp_pos_tagger_id)

test_pos_tagger_id = torch.concat(temp_test_pos_tagger_id_tensor)
test_parser_id = torch.concat(temp_test_parser_id_tensor)
torch.save(test_pos_tagger_id, "test_pos_tagger_id.pt")
torch.save(test_parser_id, "test_parser_id.pt")
# test_data.to_csv("test_data_pos_parser_explore.csv")

# For val dataset
temp_valid_pos_tagger_id_tensor = []
temp_valid_parser_id_tensor = []


for i in range(valid_data.shape[0]):
    
    # print(i)
    
    temp_valid_pos_tagger = valid_data["pos_tagger"][i]
    temp_valid_parser = valid_data["parser"][i]
    # print(temp_train_parser)
    
    temp_valid_pos_tagger_id = []
    temp_valid_parser_id = []
    temp_valid_pos_tagger_id_str = ""
    temp_valid_parser_id_str = ""
    
    # print(temp_pos_tagger)
    # print(temp_parser)
    
    temp_valid_pos_tagger_lst = temp_valid_pos_tagger.split(" ")
    temp_valid_parser_lst = temp_valid_parser.split(" ")
    
    # print(temp_train_parser_lst)
    
    if len(temp_valid_pos_tagger_lst) <= 512:
        for pt in temp_valid_pos_tagger_lst:
            
            temp_valid_pos_tagger_id.append(pos2index[pt])
            
            temp_valid_pos_tagger_id_str += pt
            temp_valid_pos_tagger_id_str += " "
            
        for pt_index in range(512 - len(temp_valid_pos_tagger_lst)):
            
            temp_valid_pos_tagger_id.append(pos2index["padding"])
            
            temp_valid_pos_tagger_id_str += "padding"
            temp_valid_pos_tagger_id_str += " "
            
        valid_data["pos_tagger_id"][i] = temp_valid_pos_tagger_id_str
            
    else:
        for pt_index in range(512):
            
            temp_valid_pos_tagger_id.append(pos2index[temp_valid_pos_tagger_lst[pt_index]])
            
            temp_valid_pos_tagger_id_str += temp_valid_pos_tagger_lst[pt_index]
            temp_valid_pos_tagger_id_str += " "
            
        valid_data["pos_tagger_id"][i] = temp_valid_pos_tagger_id_str
        
    if len(temp_valid_parser_lst) <= 512:
        # print("ok")
        for par in temp_valid_parser_lst:
            
            # print(par)
            
            temp_valid_parser_id.append(par2index[par])
            
            temp_valid_parser_id_str += pt
            temp_valid_parser_id_str += " "
            
        for pt_index in range(512 - len(temp_valid_parser_lst)):
            
            temp_valid_parser_id.append(par2index["padding"])
            
            temp_valid_parser_id_str += "padding"
            temp_valid_parser_id_str += " "
            
        valid_data["parser_id"][i] = temp_valid_parser_id_str
    
    else:
        for par_index in range(512):
            
            temp_valid_parser_id.append(par2index[temp_valid_parser_lst[pt_index]])
            
            temp_valid_parser_id_str += temp_valid_parser_lst[pt_index]
            temp_valid_parser_id_str += " "
            
        valid_data["pos_tagger_id"][i] = temp_valid_parser_id_str
    
    temp_valid_pos_tagger_id_tensor.append(torch.tensor(temp_valid_pos_tagger_id).unsqueeze(0))
    temp_valid_parser_id_tensor.append(torch.tensor(temp_valid_parser_id).unsqueeze(0))
        
    # print(temp_pos_tagger_id)

valid_pos_tagger_id = torch.concat(temp_valid_pos_tagger_id_tensor)
valid_parser_id = torch.concat(temp_valid_parser_id_tensor)
torch.save(valid_pos_tagger_id, "valid_pos_tagger_id.pt")
torch.save(valid_parser_id, "valid_parser_id.pt")
# valid_data.to_csv("valid_data_pos_parser_explore.csv")

# print(train_pos_tagger_id.shape)
# print(test_pos_tagger_id.shape)
# print(valid_pos_tagger_id.shape)

y_train = torch.tensor(np.array(y_train).astype(float))
torch_trainset = Data.TensorDataset(X_train, train_pos_tagger_id, train_parser_id, y_train)

train_loader = Data.DataLoader(
dataset=torch_trainset,    
batch_size = 512,     
shuffle=True,          
num_workers=2,              
drop_last = True,
)

y_test = torch.tensor(np.array(y_test).astype(float))
torch_testset = Data.TensorDataset(X_test, test_pos_tagger_id, test_parser_id, y_test)

test_loader = Data.DataLoader(
dataset=torch_testset,    
batch_size = 512,     
shuffle=True,          
num_workers=2,              
drop_last = True,
)

y_valid = torch.tensor(np.array(y_valid).astype(float))
torch_validset = Data.TensorDataset(X_valid, valid_pos_tagger_id, valid_parser_id, y_valid)

valid_loader = Data.DataLoader(
dataset=torch_validset,    
batch_size = 512,     
shuffle=True,          
num_workers=2,              
drop_last = True,
)

# print(X_valid.shape)
# print(valid_pos_tagger_id.shape)
# print(valid_parser_id.shape)
# print(y_valid.shape)
# print(len(pos2index))
# print(len(par2index))

# train_pos_tagger_id_ex = torch.randint(10, (300, 512))
# train_parser_id_ex = torch.randint(10, (300, 512))

# pos_tagger_embedding = nn.Embedding(num_embeddings = 37, embedding_dim = 16)
# em = pos_tagger_embedding(train_pos_tagger_id_ex)
# print(em.shape)

def print_grad(grad):
    print(grad)

class TextCNN(nn.Module):

    def __init__(self, embedding_dim_input, embedding_dim_pt, embedding_dim_par, embedded_dim_pt, embedded_dim_par, num_classes, kernel_sizes=[3, 4, 5], num_filters=100):
        super(TextCNN, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pt_embedding = nn.Embedding(num_embeddings = embedding_dim_pt, embedding_dim = embedded_dim_pt)
        self.par_embedding = nn.Embedding(num_embeddings = embedding_dim_par, embedding_dim = embedded_dim_par)
        self.convs_input = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim_input)) for k in kernel_sizes
        ])
        self.convs_pt = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedded_dim_pt)) for k in kernel_sizes
        ])
        self.convs_par = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedded_dim_par)) for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters * 3, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, pt, par):
        # x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
        x_pt = self.pt_embedding(pt)
        x_pt = x_pt.unsqueeze(1)
        x_par = self.par_embedding(par)
        x_par = x_par.unsqueeze(1)
        # Apply convolutions and max-pooling
        conv_outputs_input = []
        conv_outputs_pt = []
        conv_outputs_par = []
        for conv in self.convs_input:
            conv_output_input = conv(x)  # [batch_size, num_filters, seq_len - kernel_size + 1, 1]
            activation = F.relu(conv_output_input)
            max_pooled = F.max_pool2d(activation, (conv_output_input.shape[2], 1))  # [batch_size, num_filters, 1, 1]
            conv_outputs_input.append(max_pooled.squeeze(-1).squeeze(-1))  # [batch_size, num_filters]
            
        for conv in self.convs_pt:
            conv_output_pt = conv(x_pt)  # [batch_size, num_filters, seq_len - kernel_size + 1, 1]
            activation = F.relu(conv_output_pt)
            max_pooled = F.max_pool2d(activation, (conv_output_pt.shape[2], 1))  # [batch_size, num_filters, 1, 1]
            conv_outputs_pt.append(max_pooled.squeeze(-1).squeeze(-1))  # [batch_size, num_filters]
            
        for conv in self.convs_par:
            conv_output_par = conv(x_par)  # [batch_size, num_filters, seq_len - kernel_size + 1, 1]
            activation = F.relu(conv_output_par)
            max_pooled = F.max_pool2d(activation, (conv_output_par.shape[2], 1))  # [batch_size, num_filters, 1, 1]
            conv_outputs_par.append(max_pooled.squeeze(-1).squeeze(-1))  # [batch_size, num_filters]
        
        conv_outputs = conv_outputs_input + conv_outputs_pt + conv_outputs_par

        # Concatenate and apply dropout
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)

        # Apply linear layer
        x = self.fc(x)
        return x
    
# pred = Text(X_train, train_pos_tagger_id_ex, train_parser_id_ex)

# print(pred.shape)

model = TextCNN(embedding_dim_input = 200, embedding_dim_pt = len(pos2index), embedded_dim_pt = 16, embedded_dim_par = 16, embedding_dim_par = len(par2index), num_classes = 4)


print("Start training!!!")
num_epoch = 64
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

  if (epoch + 1) % 10 == 0:
    for p in opt.param_groups:
      p['lr'] *= learning_rate_decay
  
  lr_list.append(opt.state_dict()['param_groups'][0]['lr'])
  
  for step, (X_train, train_pos_tagger_id, train_parser_id, y_train) in enumerate(train_loader):

    model.train()

    if torch.cuda.is_available() and gpu:
      X_train = X_train.to("cuda")
      train_pos_tagger_id = train_pos_tagger_id.to("cuda")
      train_parser_id = train_parser_id.to("cuda")
      y_train = y_train.to("cuda")
      model = model.to("cuda")
    outputs = model(X_train, train_pos_tagger_id, train_parser_id)

    # model.fc.weight.register_hook(print_grad)

    loss = loss_fn(outputs, y_train.long())
    opt.zero_grad()
    loss.backward()
    opt.step()

    print(outputs)

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
    for eval_step, (X_valid, valid_pos_tagger_id, valid_parser_id, y_valid) in enumerate(valid_loader):

      if torch.cuda.is_available() and gpu:
          X_valid = X_valid.to('cuda')
          y_valid = y_valid.to('cuda')
          valid_pos_tagger_id = valid_pos_tagger_id.to("cuda")
          valid_parser_id = valid_parser_id.to("cuda")
          
      eval_loss = 0
      eval_outputs = model(X_valid, valid_pos_tagger_id, valid_parser_id)
      temp_eval_loss = loss_fn(eval_outputs, y_valid.long())
      eval_loss += temp_eval_loss

      if eval_loss < min_loss:
          print("original loss: {}, new loss: {}".format(min_loss, eval_loss))
          min_loss = eval_loss
          torch.save(model.state_dict(), "model_multi_CNN.pth")
          print("save")
      valid_loss.append(eval_loss.item())

plt.figure(figsize=(10, 6))
plt.plot(train_loss)
plt.xlabel("batches")
plt.ylabel("train_loss")
plt.title("train loss v.s. epochs")
plt.savefig("train_loss_multi_CNN.jpg")

plt.figure(figsize=(10, 6))
plt.plot(train_acc)
plt.xlabel("batches")
plt.ylabel("train_accuracy")
plt.title("train accuracy v.s. epochs")
plt.savefig("train_accuracy_multi_CNN.jpg")

# plt.figure(figsize=(10, 6))
# plt.plot(train_recall)
# plt.xlabel("epochs")
# plt.ylabel("train_recall")
# plt.title("train recall v.s. epochs")
# plt.savefig("train_recall_multi_CNN.jpg")

plt.figure(figsize=(10, 6))
plt.plot(valid_loss)
plt.xlabel("batches")
plt.ylabel("valid loss")
plt.title("valid loss v.s. epochs")
plt.savefig("valid_loss_multi_CNN.jpg")

plt.figure(figsize=(10, 6))
plt.plot(lr_list)
plt.xlabel("batches")
plt.ylabel("learning_rate")
plt.title("learning_rate v.s. epochs")
plt.savefig("learning_rate_multi_CNN.jpg")

# Calculate Metrics
model.eval()
test_loss = 0
test_acc = 0
test_recall = 0
num_true_pred = 0
num_true_ner = 0
test_num = 0
ner_num = 0

for test_step, (X_test, valid_pos_tagger_id, valid_parser_id, y_test) in enumerate(test_loader):

  if torch.cuda.is_available() and gpu:
    X_test = X_test.to('cuda')
    y_test = y_test.to('cuda')
    valid_pos_tagger_id = valid_pos_tagger_id.to('cuda')
    valid_parser_id = valid_parser_id.to('cuda')

  test_outputs = model(X_test, valid_pos_tagger_id, valid_parser_id)
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
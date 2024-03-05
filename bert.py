import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AlbertTokenizer, AlbertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from train_utils import *

# bert_name = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(bert_name)
# bert = BertModel.from_pretrained(bert_name)
bert_name = "albert-base-v2"
tokenizer = AlbertTokenizer.from_pretrained(bert_name)
bert = AlbertModel.from_pretrained(bert_name)
    
## Load and split data
df_train = pd.read_csv("train.csv", low_memory=False)
df_train.columns
X = df_train.drop("price", axis=1)
y = df_train['price']
X_train, X_val, y_train, y_val = train_test_split(X, y.values, test_size=0.2)
X, y = shuffle(X, y) ## shuffle dataset, maybe it helps?
X_submission = pd.read_csv("test.csv", low_memory=False)
MAX_TOKEN_SIZE_BERT = 512

class custom_bert_regressor(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(768, 1)
        def MLP(i, h, o):
            mlp = nn.Sequential(
                nn.Linear(i, h),
                nn.GELU(),
                nn.Linear(h, o)
            )
            return mlp
        self.mlp = MLP(768, 100, 1)

    def forward(self, x, mask):
        # with torch.no_grad():
        bert_output = self.bert(input_ids=x, attention_mask=mask)
        cls = bert_output.last_hidden_state[:, 0, :]
        out = self.mlp(cls)
#         out = self.linear(cls)
        return cls, out
    
text_attributes = ["name"]
# text_attributes = ["name", "summary", "space", "description", "neighborhood_overview", "notes", "transit", "access", "interaction", "house_rules", "host_about"]

for i, attribute in enumerate(text_attributes):
    ## transform data into token_ids
    with torch.no_grad():
        bert_all_tokens = tokenizer(
            X[attribute].fillna("UNK").to_list(),
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            truncation=True,
            max_length=512,
            )
        bert_submission_tokens = tokenizer(
            X_submission[attribute].fillna("UNK").to_list(),
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            truncation=True,
            max_length=512,
              )
        
    batch_size = 32
    epochs = 1 
    device = torch.device("cuda")
    tensor_dataset = TensorDataset(bert_all_tokens.input_ids, bert_all_tokens.attention_mask, torch.tensor(y).float())
    train_loader = DataLoader(tensor_dataset, batch_size=batch_size)

    model = custom_bert_regressor(bert).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.train()

    ## Train Bert model
    loss = 0 
    for i in range(epochs):
        for i, (x, mask, target) in enumerate(train_loader):
            x, mask, target = x.to(device), mask.to(device), target.to(device)
            _, out = model(x, mask)#[1].squeeze()#.float()
            loss = F.mse_loss(out.squeeze(), target)
            print(loss.item())
            loss.backward()
        #     # clip_grad_norm_(model.parameters(), 0.1)
            clip_grad_value_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
        print("epoch finished")
    

    with torch.no_grad():
        tensor_dataset = TensorDataset(bert_submission_tokens.input_ids, bert_submission_tokens.attention_mask)
        submission_loader = DataLoader(tensor_dataset, batch_size=batch_size)
                
        final_train_prediction = torch.empty(len(train_loader), batch_size)
        model.eval()        
        for x, mask, target in train_loader:
            x, mask = x.to(device), mask.to(device),
            _, out = model(x, mask)
            final_train_prediction[i, :out.size(0)] = out.squeeze()

        final_train_prediction = final_train_prediction.view(-1)[:-7]
        final_prediction = torch.empty(len(train_loader), batch_size)
        model.eval()        
        for x, mask in submission_loader:
            x, mask = x.to(device), mask.to(device),
            _, out = model(x, mask)
            final_prediction[i, :out.size(0)] = out.squeeze()

        final_submission_cut = final_prediction.view(-1)[:-7]
        final_submission_dict = {"train": final_train_prediction, "submission": final_submission_cut}
        torch.save(final_submission_dict, f"bert_{attribute}.pt")
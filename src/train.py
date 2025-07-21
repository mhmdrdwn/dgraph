from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, auc, roc_auc_score
from tqdm import tqdm
import torch
import numpy as np


            
def print_acc(model, data_iter):
    outs= []
    ys = []
    
    model.eval()
    with torch.no_grad():
        for X, A, y in data_iter:
            y = torch.argmax(y, -1)
            out = model(X, A)
            out = torch.exp(out)
            outs.extend(out.cpu().detach().numpy())
            ys.extend(y.cpu().detach().numpy())
    
    outs = np.array(outs)
    ys = np.array(ys)
    outs = np.argmax(outs, -1)

    print("accuracy:", accuracy_score(outs, ys),
          "f1 score:", f1_score(outs, ys),
          "precision:",precision_score(outs, ys),
          "recall:", recall_score(outs, ys),
          "confusion matrix:", confusion_matrix(outs, ys))

    metrics = [accuracy_score(outs, ys), f1_score(outs, ys), 
               precision_score(outs, ys), recall_score(outs, ys),
               confusion_matrix(outs, ys)]
    
    return metrics

    
def train_model(model, num_epochs, data_iter, lr=1e-4):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    
    model.train()
    for epoch in tqdm(range(num_epochs)): 
        model.train()
        losses = 0
        for idx, (X, A, y) in enumerate(data_iter):
            y = torch.argmax(y, -1)
            optimizer.zero_grad()
            out = model(X, A)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses += loss
    return model
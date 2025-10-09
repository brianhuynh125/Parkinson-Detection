import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from model import HybridCNN_RNN_MKL
from dataprocessing import make_loaders
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
df = pd.read_csv('dataset_manifest.csv')

EPOCHS = 25
BATCH_SIZE = 8
LR = 3e-4
WD = 1e-5
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

fold_metrics = []

for fold in range(5):
    
    print(f"\n===== Fold {fold+1}/5 =====")
    train_loader, val_loader = make_loaders(df, kfold_idx=fold, batch_size=BATCH_SIZE)

    model = HybridCNN_RNN_MKL().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_auc, best_state = 0.0, None

    for epoch in range(EPOCHS):

        
        model.train()
        running_loss = 0
        for spec, acoustics, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            spec, label = spec.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(spec)
            out = out.view(-1)
            label = label.view(-1)
            loss = criterion(out, label)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()


        model.eval()
        preds, probs, labels = [], [], []
        cnt = 1
        with torch.no_grad():
            for spec, acoustics, label in val_loader:
                print("order of val load:", cnt)
                # out = model(spec.to(device)).cpu().squeeze()
                # print("original model output", out)
                # out_np = out.numpy().ravel()  # flatten to always 1D
                # logits = model(spec.to(device)).cpu().squeeze()
                # print("Model output logits", logits)
                # prob = torch.sigmoid(logits).numpy().ravel()
                # for i in prob:
                #     probs.append(i)
                # print("probss of being 1", probs)
                # (probs > 0.5).astype(float).tolist():
                #     preds.append(i)
                logits = model(spec.to(device)).cpu().squeeze()
                prob = torch.sigmoid(logits).numpy().ravel()
                probs.extend(prob.tolist())                    # add all probabilities to list
                pred = (prob > 0.5).astype(float).tolist()     # get binary predictions
                preds.extend(pred)  
                
                
                # probs.extend(out_np.tolist())
                # preds.extend((out_np > 0.5).astype(int).tolist())
                labels.extend(label.numpy().ravel().tolist())
                
                cnt += 1
        print("current preds list", preds)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        print(f"Epoch {epoch+1} | Loss={running_loss/len(train_loader):.4f} | ACC={acc:.3f} | F1={f1:.3f} | AUC={auc:.3f}")

        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()

    # Save best model for this fold
    torch.save(best_state, f"{SAVE_DIR}/fold_{fold+1}_best.pth")
    fold_metrics.append((acc, f1, auc))
    print(f"Best AUC for Fold {fold+1}: {best_auc:.3f}")

# ===== Summary =====
accs, f1s, aucs = np.array(fold_metrics).T
print("\n===== 5-Fold Cross Validation Results =====")
print(f"Accuracy: {accs.mean():.3f} ± {accs.std():.3f}")
print(f"F1 Score: {f1s.mean():.3f} ± {f1s.std():.3f}")
print(f"AUC:      {aucs.mean():.3f} ± {aucs.std():.3f}")

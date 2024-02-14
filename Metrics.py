"""
reference: https://blog.csdn.net/u010505915/article/details/106450150
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt


def multi_auc(y, y_pred): 
    n_classes = len(np.unique(y))
    fpr = dict()
    tpr = dict()

    for i in range(n_classes):
        y_binary = np.array([1 if label == i else 0 for label in y])
        fpr[i], tpr[i], _ = roc_curve(y_binary, y_pred[:, i])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr

    plt.clf()
    plt.plot(fpr["macro"], tpr["macro"])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
        
    return auc(fpr["macro"], tpr["macro"])


def check_metrics(train_loader, val_loader, model, flag):
    model.eval()

    train_probs = []
    train_predictions = []
    train_labels = []
    val_probs = []
    val_predictions = []
    val_labels = []

    with torch.no_grad():
        for x, y in train_loader:
            x = x.squeeze(1)

            probs = model(x)
            _, predictions = probs.max(1)

            train_probs.extend(probs.numpy())
            train_predictions.extend(predictions.numpy())
            train_labels.extend(y.numpy())

        for x, y in val_loader:
            x = x.squeeze(1)

            probs = model(x)
            _, predictions = probs.max(1)

            val_probs.extend(probs.numpy())
            val_predictions.extend(predictions.numpy())
            val_labels.extend(y.numpy())

    print(f"Train       Accuracy: {accuracy_score(train_labels, train_predictions):.2f}")
    print(f"Train       Precision: {precision_score(train_labels, train_predictions, average='macro', zero_division=0):.2f}")
    print(f"Train       Recall: {recall_score(train_labels, train_predictions, average='macro'):.2f}")
    print(f"Train       F1-score: {f1_score(train_labels, train_predictions, average='macro'):.2f}")
    score = multi_auc(train_labels, np.vstack(train_probs))
    print(f"Train       AUC-score: {score:.2f}")
    if flag:
        plt.title(f"Train Set ROC Curve")
        plt.show()
    
    print(f"\nValidation  Accuracy: {accuracy_score(val_labels, val_predictions):.2f}")
    print(f"Validation  Precision: {precision_score(val_labels, val_predictions, average='macro', zero_division=0):.2f}")
    print(f"Validation  Recall: {recall_score(val_labels, val_predictions, average='macro'):.2f}")
    print(f"Validation  F1-score: {f1_score(val_labels, val_predictions, average='macro'):.2f}")
    score = multi_auc(val_labels, np.vstack(val_probs))
    print(f"Validation  AUC-score: {score:.2f}")
    if flag:
        plt.title(f'Validation Set ROC Curve')
        plt.show()

    model.train()
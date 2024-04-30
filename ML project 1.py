#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import yfinance as yf 


# In[2]:


df = yf.Ticker("^GSPC")


# In[24]:


df = yf.download("SPY", start="2015-01-01", end="2023-12-31")


# In[4]:


print(df.head())


# In[5]:


print(df.describe())


# In[6]:


df.plot.line(y="Close" , use_index=True)


# In[7]:


df["Tomorrow"] = df["Close"].shift(-1)


# In[8]:


df


# In[9]:


df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)


# In[10]:


df


# In[11]:


df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)


# In[12]:


df


# In[13]:


df['Signal'] = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)


# In[14]:


df = df[:-1]


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(df.drop(['Signal', 'Tomorrow'], axis=1), df['Signal'], test_size=0.2, shuffle=False)


# In[16]:


models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'RF': RandomForestClassifier(n_estimators=100, random_state=42),
    'GB': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'XGB': XGBClassifier(n_estimators=100, random_state=42)
}


# In[19]:


y_train_mapped = np.where(y_train == -1, 0, 1)
y_test_mapped = np.where(y_test == -1, 0, 1)


# In[20]:


results = {} # To store the results
for name, model in models.items():
    # Fit the model on the training set with mapped classes
    model.fit(X_train, y_train_mapped)
    # Predict on the testing set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    # Compute the metrics
    acc = accuracy_score(y_test_mapped, y_pred)
    prec = precision_score(y_test_mapped, y_pred)
    rec = recall_score(y_test_mapped, y_pred)
    f1 = f1_score(y_test_mapped, y_pred)
    cm = confusion_matrix(y_test_mapped, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test_mapped, y_prob)
    roc_auc = auc(fpr, tpr)
    # Store the results
    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1,
        'Confusion Matrix': cm,
        'FPR': fpr,
        'TPR': tpr,
        'ROC AUC': roc_auc
    }


# In[21]:


for name, result in results.items():
    print(f'Model: {name}')
    print(f'Accuracy: {result["Accuracy"]:.4f}')
    print(f'Precision: {result["Precision"]:.4f}')
    print(f'Recall: {result["Recall"]:.4f}')
    print(f'F1-score: {result["F1-score"]:.4f}')
    print(f'Confusion Matrix:\n{result["Confusion Matrix"]}')
    print(f'ROC AUC: {result["ROC AUC"]:.4f}')
    print('-'*20)


# In[22]:


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Confusion Matrices for Strategy 1')
for i, (name, result) in enumerate(results.items()):
    ax = axes.flatten()[i]
    ax.imshow(result['Confusion Matrix'], cmap='Blues')
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    for j in range(2):
        for k in range(2):
            ax.text(k, j, result['Confusion Matrix'][j, k], ha='center', va='center', color='black')
axes.flatten()[-1].axis('off')
plt.show()


# In[40]:


fig, ax = plt.subplots(figsize=(10, 10))
ax.plot([0, 1], [0, 1], 'k--')
ax.set_title('ROC Curves for Strategy 1')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
for name, result in results.items():
    ax.plot(result['FPR'], result['TPR'], label=f'{name} (AUC = {result["ROC AUC"]:.4f})')
ax.legend(loc='lower right')
plt.show()


# In[26]:


df.loc[:, 'MA50'] = df['Close'].rolling(50).mean()
df.loc[:, 'MA200'] = df['Close'].rolling(200).mean()


# In[27]:


df = df.dropna()


# In[30]:


df.loc[:, 'Signal'] = np.where(df['MA50'] > df['MA200'], 1, -1)


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Signal', axis=1), df['Signal'], test_size=0.2, shuffle=False)


# In[32]:


models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'RF': RandomForestClassifier(n_estimators=100, random_state=42),
    'GB': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'XGB': XGBClassifier(n_estimators=100, random_state=42)
}


# In[33]:


y_train_mapped = np.where(y_train == -1, 0, 1)
y_test_mapped = np.where(y_test == -1, 0, 1)


# In[34]:


results = {} # To store the results
for name, model in models.items():
    # Fit the model on the training set
    model.fit(X_train, y_train_mapped)
    # Predict on the testing set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    # Compute the metrics
    acc = accuracy_score(y_test_mapped, y_pred)
    prec = precision_score(y_test_mapped, y_pred)
    rec = recall_score(y_test_mapped, y_pred)
    f1 = f1_score(y_test_mapped, y_pred)
    cm = confusion_matrix(y_test_mapped, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test_mapped, y_prob)
    roc_auc = auc(fpr, tpr)
    # Store the results
    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1,
        'Confusion Matrix': cm,
        'FPR': fpr,
        'TPR': tpr,
        'ROC AUC': roc_auc
    }


# In[35]:


for name, result in results.items():
    print(f'Model: {name}')
    print(f'Accuracy: {result["Accuracy"]:.4f}')
    print(f'Precision: {result["Precision"]:.4f}')
    print(f'Recall: {result["Recall"]:.4f}')


# In[36]:


fig, ax = plt.subplots(figsize=(10, 10))
ax.plot([0, 1], [0, 1], 'k--')
ax.set_title('ROC Curves for Strategy 2')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
for name, result in results.items():
    ax.plot(result['FPR'], result['TPR'], label=f'{name} (AUC = {result["ROC AUC"]:.4f})')
ax.legend(loc='lower right')
plt.show()


# In[37]:


plt.figure(figsize=(10, 10))
plt.plot(df['Close'], label='Close Price')
plt.plot(df['MA50'], label='50-day Moving Average')
plt.plot(df['MA200'], label='200-day Moving Average')
plt.title('S&P 500 Index Stock with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[38]:


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Confusion Matrices for Strategy 2')
for i, (name, result) in enumerate(results.items()):
    ax = axes.flatten()[i]
    ax.imshow(result['Confusion Matrix'], cmap='Blues')
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    for j in range(2):
        for k in range(2):
            ax.text(k, j, result['Confusion Matrix'][j, k], ha='center', va='center', color='black')
axes.flatten()[-1].axis('off')
plt.show()


# In[ ]:





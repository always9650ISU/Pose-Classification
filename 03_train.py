from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import pandas as pd 
import pickle

FolderPath = './train'


if not os.path.exists(FolderPath):
    os.makedirs(FolderPath)

for foldername in os.listdir(FolderPath):
# Model params
    data_dir = os.path.join(FolderPath, foldername, 'csv_out')
    model_path = os.path.join(FolderPath, foldername, 'params')
    print(model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    model_path = os.path.join(model_path, foldername +'.pkl')
    X = [] 
    Y = []

    idx = 0

    # Collect data
    for filename in os.listdir(data_dir):
        if not filename.endswith("csv"):
            continue
        if filename == 'unlabel.csv':
            continue
        print(filename)
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath, header=None)
        
        df = df.drop(0, axis=1)
        y = np.ones(len(df)) * idx

        X.append(df.to_numpy())
        Y.append(y.tolist())

        idx += 1

    X = np.concatenate(X, axis=None).reshape(-1, 66)
    Y = np.concatenate(Y, axis=None)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42)


    # Train model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"File {foldername}, train score: {train_score}")
    print(f"File {foldername}, test score: {test_score}")
    print("_" * 10)
    with open(model_path,'wb') as f:
        pickle.dump(model,f)

    # Test model
    '''
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    model.predict(X[0:1])
    '''
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import pandas as pd
import pickle


def main():
    """
    Training model.
    """
    # Default Folder path.
    FolderPath = './train'

    # Build model.
    for foldername in os.listdir(FolderPath):
        # Name csv and model folder.
        data_dir = os.path.join(FolderPath, foldername, 'csv_out')
        model_path = os.path.join(FolderPath, foldername, 'params')

        # Auto create directory if not exists.
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Name model path.
        model_path = os.path.join(model_path, foldername + '.pkl')

        # Create keypoints ,label list for collecting.
        X = []  # For keypoints
        Y = []  # for Label
        idx = 0

        # Read csv file and label.
        for filename in os.listdir(data_dir):
            # Input only support cvs file.
            if not filename.endswith("csv"):
                continue
            # Pass unlabel file.
            if filename == 'unlabel.csv':
                continue

            print(filename)
            # Get filepath.
            filepath = os.path.join(data_dir, filename)
            # Reading csv file using dataframe format.
            df = pd.read_csv(filepath, header=None)
            # Remove index column.
            df = df.drop(0, axis=1)
            # Label by read directory order.
            y = np.ones(len(df)) * idx
            # Add file to x,y list.
            X.append(df.to_numpy())
            Y.append(y.tolist())

            idx += 1
        # Concate all keypoionts and labels
        # X(keypoints) shape like (number of images, keypoints)
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

        # Save model.
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)


if __name__ == "__main__":
    main()

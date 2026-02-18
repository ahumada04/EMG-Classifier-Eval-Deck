import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

RAW_DATA_PATH = os.path.join("data", "raw")
RAND_NUM = 25

def grabRaw(folder_path=RAW_DATA_PATH):
    # Grab all subjects    
    folder_list = glob.glob(os.path.join(folder_path, "*/"))
    full_features = []
    full_lables = []
    for folder in folder_list:
        # Grab emg data files
        file_list = glob.glob(os.path.join(folder, "*.txt"))
        for file in file_list:
            # Load data: Time (0), Channels (1-8), Class (9)
            df = pd.read_csv(file, sep='\s+')
            
            # 1. Feature Extraction: (1-8)
            features = df.iloc[:, 1:9].values 
            
            # 2. Class Extraction: (9)
            labels = df.iloc[:, 9].values
            
            # 3. Clean Data: Remove unmarked data (class == '0')
            valid_indices = (labels != 0)
            features = features[valid_indices]
            labels = labels[valid_indices]
            
            # Shift labels 1-7 -> 0-6 for NN compatibility
            labels = labels - 1
            
            # append to full list of features & labels
            full_features.extend(features)
            full_lables.extend(labels)
    return full_features, full_lables


def preProccess(X_raw, y_raw):
    scaler = StandardScaler()
    X_raw = scaler.fit_transform(X_raw)

    # 1. Combine into a temporary DataFrame to keep rows aligned
    temp_df = pd.DataFrame(X_raw)
    temp_df['label'] = y_raw

    # 2. Drop any row that has a NaN in either the features or the label
    clean_df = temp_df.dropna()

    # 3. Extract back into X and y
    X_all = clean_df.drop(columns=['label']).values
    y_all = clean_df['label'].values


    return X_all, y_all
# creates perfect windows
# moving forward may be good to do majority rule instead
def createWindows(X, y, window_size=64, stride=1):
    X_win = []
    y_win = []
    for i in range(0, len(y) - window_size + 1, stride):
        X_temp_window = X[i : i + window_size] 
        y_temp_window = y[i : i + window_size] 
        perfect, jump_val = isPure_window(y_temp_window)

        if not perfect:
            # INCOMPLETE, not needed with data this large.
            # might be helpful with FT on user data
            # jump by index instead of stride to not waste data 
            continue
        
        # transpose so shape becomes (8, 64) instead of (64, 8)
        X_win.append(X_temp_window.T) 
        y_win.append(y[i])
        
    return np.array(X_win), np.array(y_win)


def isPure_window(y_win):
    prev = y_win[0]
    for i in range(len(y_win)):
        if y_win[i] != prev:
            return False, i
        else:
            prev = y_win[i]
    
    return True, 0  


def createTestLoader(X_win, y_win, split=0.2):
    _, X_test, _, y_test = train_test_split(
        X_win,
        y_win,
        test_size=split,
        random_state=RAND_NUM,
        stratify=y_win,
    )

    plt.hist(y_test, bins=range(8), align='left', rwidth=0.8)
    plt.xticks(range(7), ['Rest', 'Fist', 'Flex', 'Exten', 'Radial', 'Ulnar', 'Palm'])
    plt.title("Distribution of Gesture Classes")
    plt.ylabel("Number of TEST Window Samples")
    plt.show()

    test_data_tensor = TensorDataset(
        torch.from_numpy(X_test).float(), 
        torch.from_numpy(y_test).long()
    )
    test_loader = DataLoader(
        test_data_tensor, 
        batch_size=32, 
        shuffle=True
    )

    return test_loader


def main(split=0.2, window_size=64):
    print("------------------Collecting Data------------------")
    X_raw, y_raw = grabRaw()
    print("-------------------Pre-Processing------------------")
    X_raw, y_raw = preProccess(X_raw, y_raw)
    print("------------------Creating Windows-----------------")
    print(f"Window Size: {window_size}")
    X_win, y_win = createWindows(X_raw, y_raw, window_size=window_size)
    plt.hist(y_win, bins=range(8), align='left', rwidth=0.8)
    plt.xticks(range(7), ['Rest', 'Fist', 'Flex', 'Exten', 'Radial', 'Ulnar', 'Palm'])
    plt.title("Distribution of Gesture Classes")
    plt.ylabel("Number of TOTAL Window Samples")
    plt.show()
    print("----------------Creating TestLoader----------------")
    # can update split value to decide what % of data dedicated to test eval
    print(f"Test Split: {split}")
    test_loader = createTestLoader(X_win, y_win, split=split)
    loader_dir = os.path.join("data", "testLoaders")
    os.makedirs(loader_dir, exist_ok=True) 
    test_loader_path = os.path.join(loader_dir, f"testLoader_{split}_{window_size}.pth")
    try:
        torch.save(test_loader, test_loader_path)
        print(f"Successfully created TestLoader at \n{test_loader_path}")

    except PermissionError:
        print(f"Error: Permission denied. Close the file if it's open in another program.")

    except Exception as e:
        print("Failed saving TestLoader")
        print(f"Error Details: {e}")


if __name__ == "__main__":
    main()

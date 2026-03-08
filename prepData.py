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
LOADER_PATH = os.path.join("data", "testLoaders")
RAND_NUM = 25

def grab_raw(folder_path=RAW_DATA_PATH):
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

            # 3. Clean Data: Remove unmarked data and palm data
            valid_indices = (labels != 0) & (labels!= 7)
            features = features[valid_indices]
            labels = labels[valid_indices]
            
            # Shift labels 1-6 -> 0-5 for NN compatibility
            labels = labels - 1
            
            # append to full list of features & labels
            full_features.extend(features)
            full_lables.extend(labels)
    return full_features, full_lables


def pre_proccess(X_raw, y_raw):
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


def create_windows_pure(X, y, window_size, stride):
    X_win = []
    y_win = []
    loss_data = 0
    for i in range(0, len(y) - window_size + 1, stride):
        X_temp_window = X[i : i + window_size] 
        y_temp_window = y[i : i + window_size] 
        perfect, jump_val = isPure_window(y_temp_window)
        
        if not perfect:
            loss_data += 1
            # INCOMPLETE, not needed with data this large.
            # might be helpful with FT on user data
            # jump by index instead of stride to not waste data 
            continue
        
        # We transpose so shape becomes (8, WINDOW_SIZE) instead of (WINDOW_SIZE, 8)
        X_win.append(X_temp_window.T) 
        y_win.append(y[i])
    print(f"Data Loss to impure windows: {loss_data}")
    return np.array(X_win), np.array(y_win)


def isPure_window(y_win):
    prev = y_win[0]
    for i in range(len(y_win)):
        if y_win[i] != prev:
            return False, i
        else:
            prev = y_win[i]
    
    return True, 0  


def create_windows_majority(X, y, window_size, stride):
    X_win = []
    y_win = []
    lost_data = 0
    for i in range(0, len(y) - window_size + 1, stride):
        X_temp_window = X[i : i + window_size] 
        y_temp_window = y[i : i + window_size] 
        label = majorityClass_window(y_temp_window)
        
        if label is None:  
            lost_data += 1
            continue
        
        X_win.append(X_temp_window.T) 
        y_win.append(label)

    print(f"Data Loss to impure windows: {lost_data}")
    return np.array(X_win), np.array(y_win)


def majorityClass_window(y_win):
    counts = np.zeros(7)
    size = len(y_win)

    for val in y_win:
        counts[int(val)] += 1

    majLabel_idx = counts.argmax()

    if counts[majLabel_idx] > size / 2:
        return majLabel_idx  
    
    return None  


def createTestLoader(X_test, y_test):
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


def main(window_size, stride, window_type):
    print("------------------Collecting Data------------------")
    X_test_raw, y_test_raw = grab_raw()
    print("-------------------Pre-Processing------------------")
    X_test_raw, y_test_raw = pre_proccess(X_test_raw, y_test_raw)
    print("------------------Creating Windows-----------------")
    print(f"Window Size: {window_size}")
    if window_type=='pure':
        print("Creating PURE Windowed TESTING data")
        X_test, y_test = create_windows_pure(X_test_raw, y_test_raw, window_size=window_size, stride=stride)
    elif window_type=='majority':
        print("Creating MAJORITY Windowed TESTING data")
        X_test, y_test = create_windows_majority(X_test_raw, y_test_raw, window_size=window_size, stride=stride)
    else: 
        print("NOT a valid window type.")
        print("Please set WINDOW_TYPE to \'pure\' OR \'majority\'")
    plt.hist(y_test, bins=range(7), align='left', rwidth=0.8)
    plt.xticks(range(6), ['Rest', 'Fist', 'Flex', 'Exten', 'Radial', 'Ulnar'])
    plt.title("Distribution of Gesture Classes")
    plt.ylabel("Number of TOTAL Window Samples")
    plt.show()
    print("----------------Creating TestLoader----------------")
    test_loader = createTestLoader(X_test, y_test)
    test_loader_path = os.path.join(LOADER_PATH, f"testLoader_{window_type[0].upper()}W_W{window_size}_S{stride}.pth")
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

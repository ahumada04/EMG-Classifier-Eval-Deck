import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report
import prepData

LOADER_PATH = os.path.join("data", "testLoaders")
GESTURE_NAMES = ['Rest', 'Fist', 'Flexion', 'Extension', 'Radial', 'Ulnar']

class Eval:
    def __init__(self, model, writer, window_size, stride, window_type="pure"):
        self.model = model
        self.writer = writer
        self.config_name = f"{window_type[0].upper()}W_W{window_size}_S{stride}.pth"

        test_loader_path = os.path.join(LOADER_PATH, f"testLoader_{self.config_name}")
        if not os.path.exists(test_loader_path):
            print("Creating Test Loader...")
            prepData.main(window_size, stride, window_type)
        
        self.test_loader = torch.load(test_loader_path)
        self.y_true, self.y_pred = self.get_pred()
        # since this isn't gonna be written to tensorboard logs anyways
        print(classification_report(self.y_true, self.y_pred, target_names=GESTURE_NAMES))


    def get_pred(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad(): 
            for inputs, labels in self.test_loader:
                # Forward pass
                outputs = self.model(inputs)
                
                # Get highest prob predicted class
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
        return np.array(all_labels), np.array(all_preds)
    
    # Matthews Correlation Coefficient
    def write_mcc(self):
        mcc = matthews_corrcoef(self.y_true, self.y_pred)  # ranges -1 to 1, 1 being perfect
        print(f"MCC: {mcc:.4f}")
        self.writer.add_scalar(f'MCC/{self.model.__class__.__name__}_{self.config_name}', mcc)


    # Confusion Matrix
    def write_cm(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        cm_perc = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :] * 100
        
        fig = plt.figure(figsize=(10, 7))
        sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Blues', 
                    xticklabels=GESTURE_NAMES, 
                    yticklabels=GESTURE_NAMES)
        plt.xlabel('Predicted Gesture')
        plt.ylabel('Actual Gesture')
        plt.title(f'Confusion-Matrix/{self.model.__class__.__name__}_{self.config_name}')
        
        self.writer.add_figure(f'Confusion_Matrix/{self.model.__class__.__name__}_{self.config_name}', fig)

    # TODO
    def write_F1():
        pass

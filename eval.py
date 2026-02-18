import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# RENAME LATER
# Any suggestions plz...
def testPredictions(model, test_loader):
    # model.eval()  
    all_preds = []
    all_labels = []
    with torch.no_grad(): 
        for inputs, labels in test_loader:
            # forward pass
            outputs = model(inputs)
            
            # get gesture classifications
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    

    return np.array(all_labels), np.array(all_preds)


# TODO 
# 1. More meaningful data uploads
# 2. Compare against what params?
# 2a. seperate graphs for training? 
def runEval(y_true, y_pred, writer, epochs):
    # epochs = 5
    # 1. Run Evaluation
    test_acc = accuracy_score(y_true, y_pred) * 100
    print(test_acc)
    # 2. Define Gesture Names for clarity
    gesture_names = ['Rest', 'Fist', 'Flexion', 'Extension', 'Radial', 'Ulnar', 'Palm']
    
    # 3. Create Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_perc = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :] * 100
    
    # 4. Plot using Seaborn for a cuter look
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=gesture_names, 
                yticklabels=gesture_names)
    plt.xlabel('Predicted Gesture')
    plt.ylabel('Actual Gesture')
    plt.title('sEMG Gesture Classification Confusion Matrix')
    plt.show()

    # # --- NEW TENSORBOARD LOGGING STEPS ---

    # # 5. Log Confusion Matrix Plot
    writer.add_figure('Confusion_Matrix/Test', fig, global_step=epochs)

    # 6. Log Accuracy Scalar
    writer.add_scalar('Accuracy/Test', test_acc, global_step=epochs)

    # 7. (Optional) Log specific metrics from the report
    report = classification_report(y_true, y_pred, target_names=gesture_names, output_dict=True)
    writer.add_scalar('Precision/Weighted_Avg', report['weighted avg']['precision'], global_step=epochs)

    writer.flush()
    writer.close()
    plt.show()
    print(classification_report(y_true, y_pred, target_names=gesture_names))

# TODO, create main for cleaner code...
# def main():
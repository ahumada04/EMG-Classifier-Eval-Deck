import os
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import modelClasses
import eval


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PLEASE FILL IN THE FOLLOWING
# Split of data you WANT for TESTINT
# range (0, 1)
split = 0.2
# Window size used in YOUR TRAINING
window_size = 64
# Epochs used in YOUR TRAINING 
epochs = 5
# Algorithim you used in YOUR TRAINING
alg = "CNN"
# Model Class used in YOUR TRAINING
# MUST be copied into modelClasses.py file
model = modelClasses.SimpleEMGFANet()

# Shouldn't have to touch these :D
MODELNAME = f"{alg}_E{epochs}_W{window_size}"
MODELPATH = os.path.join("models", MODELNAME+".pth")
LOGPATH = os.path.join("logs", MODELNAME+"_"+datetime.datetime.now().strftime("%m%d"))
TESTLOADERPATH = os.path.join("data", "testLoaders", f"testLoader_{split}_{window_size}.pth")

# Writer needed to view data on TensorBoard :D
# If TensorBoard isn't updating, look here first D:
writer = SummaryWriter(LOGPATH)


# TODO
# 1. Actually ensure writer works...
def main():
    state_dict = torch.load(MODELPATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    test_loader = torch.load(TESTLOADERPATH)
    y_true, y_pred = eval.testPredictions(model, test_loader)
    # sets model.eval() for you :D
    eval.runEval(y_true, y_pred, writer, epochs)


if __name__ == "__main__":
    main()

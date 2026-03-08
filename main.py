import os
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import modelClasses
from eval import Eval


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PLEASE DEFINE ALL VARIABLES BELOW ACCORDING TO YOUR MODEL
# Window size used in YOUR TRAINING
window_size = 256
# Stride length used in YOUR TRAINING
stride=64
# Window type used in YOUR TRAINING, can ONLY be pure or majority
window_type='pure'
# MUST be copied into modelClasses.py file
model = modelClasses.CNN_FANet()
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Please ensure your model saved in models dir matches EXACTLY with your model class name
MODELPATH = os.path.join("models", f"{model.__class__.__name__}.pth")
LOGPATH = os.path.join("logs", "master")
TESTLOADERPATH = os.path.join("data", "testLoaders", f"testLoader_{window_type[0].upper()}W_W{window_size}_S{stride}.pth")


def main():
    state_dict = torch.load(MODELPATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    writer = SummaryWriter(LOGPATH)
    # will also print a classification report in terminal 
    eval = Eval(model, writer, window_size, stride, window_type)
    eval.write_cm()
    eval.write_mcc()


if __name__ == "__main__":
    main()

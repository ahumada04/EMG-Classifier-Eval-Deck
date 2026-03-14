import os
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchviz import make_dot
import modelClasses
from eval import Eval


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PLEASE DEFINE ALL VARIABLES BELOW ACCORDING TO YOUR MODEL
# Window size used in YOUR TRAINING
window_size = 250
# Stride length used in YOUR TRAINING
stride=25
# Window type used in YOUR TRAINING, can ONLY be pure or majority
window_type='pure'
# MUST be copied into modelClasses.py file
model = modelClasses.LSTMClassifier()
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Please ensure your model saved in models dir matches EXACTLY with your model class name
MODELPATH = os.path.join("models", f"{model.__class__.__name__}.pt")
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

    # NN Graphing, not especially helpful
    # summary(model, input_size=(1, 8, window_size))  # (batch, channels, window_size)
    # x = torch.randn(1, 8, window_size)  # dummy input (batch, channels, window_size)  
    # y = model(x)
    # make_dot(y, params=dict(model.named_parameters())).render("model_graph", format="png")

if __name__ == "__main__":
    main()

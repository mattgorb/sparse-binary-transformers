import torch
import os

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt") / 1e6))
    os.remove('tmp.pt')



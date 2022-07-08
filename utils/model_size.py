import torch
import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def print_model_size(mdl,):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt") / 1e6))
    os.remove('tmp.pt')


def memory_profile(model, iterator, device):
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            label, text = batch
            label = label.to(device)
            text = text.to(device)
            break
        with profile(activities=[ProfilerActivity.CPU],
                     profile_memory=True, record_shapes=True) as prof:
            model(text)
        print(prof.key_averages().table())
        #predictions = model(text).squeeze(1)


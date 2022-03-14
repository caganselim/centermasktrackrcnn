import torch


"""
This script can be used to cleanup additional params that are saved in detectron2 training, optimizer, scheduler and iteration.
For creating a base checkpoint, this is an extremely crucial step, otherwise it can hurt your training process due to 
learning rate scheduling.
"""

model_path = "./V99_ytvis_base.pth"
model = torch.load(model_path)
print("Model has the following info: ", model.keys())

# Delete these: , 'optimizer', 'scheduler', 'iteration'
model.pop("optimizer")
model.pop("scheduler")
model.pop("iteration")
torch.save(model,model_path)
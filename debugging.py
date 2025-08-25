import torch

checkpoint_path = "models/m-25012018-123527.pth.tar"
checkpoint = torch.load(checkpoint_path)
print(checkpoint.keys())
print(checkpoint['state_dict'].keys())
state_dict = checkpoint['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    new_k = k.replace('.norm.1.', '.norm1.')
    new_k = new_k.replace('.norm.2.', '.norm2.')
    new_k = new_k.replace('.conv.1.', '.conv1.')
    new_k = new_k.replace('.conv.2.', '.conv2.')
    new_state_dict[new_k] = v
checkpoint['state_dict'] = new_state_dict
print(checkpoint['state_dict'].keys())

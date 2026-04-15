import torch

ckpt = torch.load('best_tuned_spa_msjf_kronos.pt', map_location='cpu')
print("Model keys:", list(ckpt.keys()))

model_state = ckpt['model']
wrapper_state = ckpt['wrapper']

print("\nWrapper shapes:")
for k, v in wrapper_state.items():
    if 'weight' in k or 'bias' in k:
        print(f"  {k}: {v.shape}")

print("\nModel shapes:")
for k, v in model_state.items():
    if 'weight' in k or 'bias' in k:
        print(f"  {k}: {v.shape}")

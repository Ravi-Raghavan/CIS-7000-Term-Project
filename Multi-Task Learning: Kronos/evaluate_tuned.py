import torch
import sys
from spa_msjf import SPAMSJF, JointLoss
from kronos_tokenizer_wrapper import KronosEncoderWrapper
from build_data_kronos import build_dataloaders
from train_kronos import evaluate, print_metrics, save_results_to_markdown

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_dl, val_dl, test_dl, meta = build_dataloaders(
    "../Finance Data",
    seq_len=60,
    horizon=1,
    batch_size=32,
)

# Best architecture inferred from checkpoint shapes
model = SPAMSJF(
    num_stocks=5,
    input_dim=320,
    private_hidden=32,
    shared_hidden=64,
    spa_dim=32,
    num_direction_classes=3,
    num_regimes=2,
    dropout=0.5, # Assume 0.5 or doesn't matter for eval
    num_heads=2, # Will match internal shapes during eval
    num_layers=2,
)

wrapper = meta["wrapper"]

# Load checkpoint
ckpt = torch.load('best_tuned_spa_msjf_kronos.pt', map_location=device)
model.load_state_dict(ckpt['model'])
wrapper.load_state_dict(ckpt['wrapper'])

model = model.to(device)
wrapper = wrapper.to(device)

print("\nRunning Evaluation on Test Set...")
test_metrics = evaluate(model, wrapper, test_dl, device)
print_metrics(test_metrics, split_name="test")

save_results_to_markdown(test_metrics, epochs=60, lr=0.0, checkpoint="best_tuned_spa_msjf_kronos.pt", version="Tuned_Colab")

import torch
from src.models.fingerprint_net import MS2FingerprintNet, SoftTanimotoLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MS2FingerprintNet(num_bins=2000, fp_dim=2048).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test batch
B = 16
spec = torch.randn(B, 2, 2000, device=device).clamp(0, 1)
precursor = torch.randn(B, 1, device=device)
adduct = torch.randint(0, 10, (B,), device=device)
ce = torch.randn(B, 1, device=device)
mode = torch.ones(B, 1, device=device)

target_fp = (torch.rand(B, 2048, device=device) > 0.95).float()

criterion_tanimoto = SoftTanimotoLoss()
criterion_bce = torch.nn.BCELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Forward
out = model(spec, precursor, adduct, ce, mode)
print("Output shape:", out.shape, "min:", out.min().item(), "max:", out.max().item())

loss_t = criterion_tanimoto(out, target_fp)
loss_bce = criterion_bce(out, target_fp)
total_loss = loss_t + 5.0 * loss_bce
print(f"Loss Tanimoto: {loss_t.item():.4f}, BCE: {loss_bce.item():.4f}, Total: {total_loss.item():.4f}")

# Backward
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
print("Backward step successful! GPU Memory allocated:", round(torch.cuda.memory_allocated() / (1024**2), 2), "MB")

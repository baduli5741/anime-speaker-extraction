# Auto-loaded at Python startup. V100 (SM 7.0) is incompatible with bundled cuDNN 9.16 (needs SM>=7.5).
# Disable cuDNN so conv ops use the native ATen path.
try:
    import torch
    torch.backends.cudnn.enabled = False
except Exception:
    pass

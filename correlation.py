# ===============================
# CPU SAFE FALLBACK CORRELATION
# ===============================

import torch
import torch.nn.functional as F

class Stream:
    ptr = None  # dummy placeholder to avoid errors

def correlate(tensorFirst, tensorSecond):
    """
    Lightweight matching approximation using normalized dot product
    Output shape: Bx1xHxW
    """
    # Normalize across channel dimension
    f1 = F.normalize(tensorFirst, dim=1)
    f2 = F.normalize(tensorSecond, dim=1)

    # Dot product along channels → correlation map
    corr = torch.sum(f1 * f2, dim=1, keepdim=True)

    # Scale to 0–1 range
    corr_min = corr.amin(dim=(2,3), keepdim=True)
    corr_max = corr.amax(dim=(2,3), keepdim=True)
    corr = (corr - corr_min) / (corr_max - corr_min + 1e-6)

    return corr

# Interface to match original module wrapping
class ModuleCorrelation(torch.nn.Module):
    def __init__(self):
        super(ModuleCorrelation, self).__init__()

    def forward(self, first, second):
        return correlate(first, second)

# Compatibility wrapper
def FunctionCorrelation(tensorFirst, tensorSecond):
    return correlate(tensorFirst, tensorSecond)

#!/usr/bin/env python
# Demo_Test.py — STRNN_final (motion-aware, time_dims=7)

import os
import cv2
import numpy as np
import torch
import torch.serialization
from collections import deque
from tqdm import tqdm
from scipy.io import savemat

# Import STRNN classes
from model import STRNN_final, STRNN_Static_Net, STRNN_SRFu_woAALSTM

# ------------------------ Paths ------------------------
input_video       = "./input_videos/myvideo.mp4"
weights_path      = "./weights/strnn-vgg16-diem_final.pth"
output_mat_path   = "./deep_saliency.mat"
output_video_path = "./saliency_heatmap.mp4"

# ------------------------ Inference settings ------------------------
# STRNN_final is trained to operate at 480×640 input with 60×80 output
time_dims = 7                       # temporal length D
io_h, io_w = 480, 640               # model input size
out_h, out_w = 60, 80               # model output size
device = torch.device("cpu")        # keep CPU to avoid CUDA env issues

# ------------------------ Video IO ------------------------
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {input_video}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
print(f"Processing {total_frames} frames...")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video_path, fourcc, fps, (io_w, io_h))

# ------------------------ Load model & checkpoint ------------------------
# Allowlist classes that may appear in pickled checkpoints
torch.serialization.add_safe_globals([STRNN_final, STRNN_Static_Net, STRNN_SRFu_woAALSTM])

# Build STRNN_final with default vgg16+dilated backbone, temporal fusion, AA-LSTM
model = STRNN_final(
    cnn_type="vgg16",
    use_dcn=True,
    use_cb=False,
    use_bn=False,
    nb_gaussian=8,
    cnn_stride=16,
    out_stride=8,
    iosize=[io_h, io_w, out_h, out_w],
    time_dims=time_dims,
    cat_type=[1, 1, 0, 0],  # ✅ static-only features
    pre_model_path=""
)


# Load weights from various possible formats
ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
if isinstance(ckpt, torch.nn.Module):
    ckpt = ckpt.state_dict()
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]
model.load_state_dict(ckpt, strict=False)
model.eval().to(device)

# ------------------------ Buffers for temporal input ------------------------
# STRNN_final expects a stack with length D+1; it internally uses x[:-1] and x[1:]
seq_len = time_dims + 1
frame_buf: deque[np.ndarray] = deque(maxlen=seq_len)

# We will produce one saliency map per step once the buffer is full.
num_outputs = max(0, total_frames - time_dims)
saliency_all = np.zeros((out_h, out_w, num_outputs), dtype=np.float32)

# Dummy center-bias tensor (unused because use_cb=False)
cb_dummy = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device=device)

# ------------------------ Main loop ------------------------
with torch.no_grad():
    out_index = 0
    for _ in tqdm(range(total_frames)):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # Resize to model input
        frame_bgr = cv2.resize(frame_bgr, (io_w, io_h), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.astype(np.float32) / 255.0
        frame_buf.append(frame_rgb)

        # Once we have D+1 frames, run one inference step
        if len(frame_buf) == seq_len:
            # Stack to [T, C, H, W] in float32
            x = np.stack(frame_buf, axis=0)                               # [T, H, W, C]
            x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()      # [T, C, H, W]
            x = x.to(device)

            # Forward: returns [D, 1, 60, 80]; take the last map to align with the newest frame
            out_maps, _ = model(x, cb_dummy)                              # out_maps: [D, 1, Hout, Wout]
            last_map = out_maps[-1, 0]                                    # [Hout, Wout]

            # Normalize to [0,1]
            pred = last_map.cpu().numpy()
            pmin, pmax = float(pred.min()), float(pred.max())
            if pmax > pmin:
                pred = (pred - pmin) / (pmax - pmin)
            else:
                pred = np.zeros_like(pred, dtype=np.float32)

            # Save compact map for MATLAB
            saliency_all[:, :, out_index] = pred

            # Visualize: upsample to io_h×io_w and apply colormap
            pred_up = cv2.resize(pred, (io_w, io_h), interpolation=cv2.INTER_LINEAR)
            heat_u8 = np.clip(pred_up * 255.0, 0, 255).astype(np.uint8)
            heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

            # Optional: overlay heatmap on the original frame for context
            overlay = cv2.addWeighted(frame_bgr, 0.35, heat_bgr, 0.65, 0.0)
            writer.write(overlay)

            out_index += 1

cap.release()
writer.release()

# ------------------------ Save MATLAB file ------------------------
# Note: number of maps = total_frames - time_dims (first 6 frames used for warm-up)
savemat(output_mat_path, {"saliency_maps": saliency_all})

print("✅ DONE!")
print(f"Saved MAT:   {output_mat_path}  (shape {saliency_all.shape})")
print(f"Saved Video: {output_video_path}  ({fps:.2f} fps, {io_w}x{io_h})")

import os
import cv2
import torch
import imageio
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from src.models import build_backbone
from src.gradcam import GradCAM
from src.utils.extract_frames import extract_frames


def process_video_with_gradcam(
    video_path: str,
    output_gif: str,
    backbone: str = "resnet50",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    fps: int = 15,
    input_size: int = 224,
):
    """
    Run Grad-CAM on every frame and save a single animated GIF of heatmaps.
    No per-frame PNG/NPY files or overlays are created.
    """

    # 1) Load frames fully in memory
    frames = extract_frames(video_path)
    total_frames = len(frames)
    print(f"🔹 Loaded {total_frames} frames from {video_path}")

    # 2) Load pretrained backbone
    model = build_backbone(backbone, pretrained=True).to(device)
    model.eval()

    # Pick the last conv2d layer automatically
    target_layer = None
    for _, module in list(model.named_modules())[::-1]:
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break
    if target_layer is None:
        raise RuntimeError("No convolutional layer found for Grad-CAM.")
    print(f"✅ Using target layer: {target_layer.__class__.__name__}")

    gradcam = GradCAM(model, target_layer)

    # Preprocessing to match ImageNet training
    preprocess = transforms.Compose([
        transforms.ToTensor(),                                        # HWC [0,255] -> CHW [0,1]
        transforms.Resize((input_size, input_size), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    heatmaps_rgb = []
    with torch.set_grad_enabled(True):
        for frame in tqdm(frames, desc="Computing Grad-CAM"):
            # BGR -> RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, W = img_rgb.shape[:2]

            # Keep original size for the output heatmap but feed normalized 224x224 to the model
            tensor_in = preprocess(img_rgb).unsqueeze(0).to(device)  # [1,3,224,224]

            cam_224 = gradcam.generate(tensor_in)                    # [224,224] in [0,1]

            # Resize CAM back to original frame resolution for visualization
            cam_full = cv2.resize(cam_224, (W, H), interpolation=cv2.INTER_CUBIC)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_full), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmaps_rgb.append(heatmap)

    # 4) Save one animated GIF
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)
    with imageio.get_writer(output_gif, mode="I", fps=fps) as writer:
        for frame in heatmaps_rgb:
            writer.append_data(frame)
    print(f"✅ Saved Grad-CAM FoV GIF → {output_gif}")



if __name__ == "__main__":
    process_video_with_gradcam(
        video_path="data/videos/myvideo.mp4",
        output_gif="outputs/fov_gradcam.gif",
        backbone="resnet50",
        device="cuda" if torch.cuda.is_available() else "cpu",
        fps=15,
        input_size=224,
    )

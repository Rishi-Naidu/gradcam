import cv2
import numpy as np

def overlay_gradcam_on_frame(frame, cam, alpha=0.5):
    """
    Blend the Grad-CAM heatmap with the original frame for visualization.

    Parameters:
        frame (np.ndarray): Original frame in BGR format.
        cam (np.ndarray): Grad-CAM map, normalized to [0, 1].
        alpha (float): Weight for heatmap overlay transparency.

    Returns:
        np.ndarray: Combined image showing attention overlaid on the frame.
    """
    # Normalize and resize Grad-CAM
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))

    # Apply color map
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Overlay heatmap on frame
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
    return overlay
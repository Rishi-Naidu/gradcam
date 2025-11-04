import torch
import torch.nn.functional as F

class GradCAM:
    """
    Minimal Grad-CAM for classification-style CNNs (e.g., ResNet, VGG).
    target_layer must be a nn.Conv2d module from the model.
    """

    def __init__(self, model, target_layer):
        self.model = model                     # keep reference
        self.model.eval()                      # BN/Dropout in eval, gradients still enabled
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Use the modern full backward hook to be future-proof
        self.hook_handles = []
        self.hook_handles.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self.hook_handles.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, inputs, output):
        # keep computation graph (no .detach())
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple; take the gradient wrt layer output
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        """
        input_tensor: shape [1, C, H, W], float32, requires grad enabled
        target_class: optional int. If None, uses argmax of model output.
        returns: np.ndarray heatmap in [0,1] with shape [H, W]
        """
        # Ensure grads flow
        for p in self.model.parameters():
            p.requires_grad_(True)
        input_tensor = input_tensor.requires_grad_(True)

        # Forward with gradients enabled
        with torch.enable_grad():
            output = self.model(input_tensor)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        # Backward on the scalar logit for target class
        self.model.zero_grad(set_to_none=True)
        loss = output[0, target_class]
        loss.backward(retain_graph=False)

        # Compute CAM
        # activations: [1, C, h, w], gradients: [1, C, h, w]
        A = self.activations
        G = self.gradients
        # Global-average-pool the gradients over spatial dims to get channel weights
        weights = G.mean(dim=(2, 3), keepdim=True)         # [1, C, 1, 1]
        cam = (weights * A).sum(dim=1, keepdim=True)       # [1, 1, h, w]
        cam = F.relu(cam)

        # Normalize to [0,1]
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)                    # [H, W]
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach().cpu().numpy()

    def __del__(self):
        # Clean up hooks
        if hasattr(self, "hook_handles"):
            for h in self.hook_handles:
                try:
                    h.remove()
                except Exception:
                    pass

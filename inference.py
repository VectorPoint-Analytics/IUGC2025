import numpy as np
import torch
import os
import re
import argparse
from PIL import Image
from models import HRNetEncoderWrapper, HRNetWithHeatmapDecoder
import albumentations as A
from albumentations.pytorch import ToTensorV2

# IMPORTANT: Input and Output Specification
# ----------------------------------------
# Input (X):
#   - A PIL.Image object (from PIL.Image.open)
#   - Represents an RGB image
#
# Output (coords):
#   - A numpy array of shape (6,) containing 3 keypoint coordinates
#   - Format: [x1, y1, x2, y2, x3, y3]
#   - Coordinates must be in the pixel space of the original input image
#   - The coordinates should represent the exact locations of the detected keypoints

class model:
    def __init__(self):
        '''
        Initialize the model
        '''
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HRNetWithHeatmapDecoder(HRNetEncoderWrapper()).to(self.device)
        dummy_output = self.model(torch.zeros(1, 1, 512, 512).to(self.device))

    def load(self, path="./"):
        '''
        Load model weights
        '''
        # Get all files in the directory
        all_files = os.listdir(path)

        # Use regex to filter files matching model pattern with .pt or .pth extension
        model_pattern = re.compile(r'.*model.*\.(?:pt|pth)$', re.IGNORECASE)
        possible_model_paths = [os.path.join(
            path, f) for f in all_files if model_pattern.match(f)]

        for model_path in possible_model_paths:
            if os.path.exists(model_path):
                print(f"Loading model: {model_path}")
                try:
                    checkpoint = torch.load(
                        model_path, map_location=self.device)
                    # Check if it's a checkpoint containing multiple components
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        # Use the model state dict instead of the entire checkpoint
                        self.model.load_state_dict(
                            checkpoint["model_state_dict"])
                        print(f"Successfully loaded model_state_dict from checkpoint")
                    else:
                        # Try to load directly, assuming it's a simple model state dictionary
                        self.model.load_state_dict(checkpoint)
                    return self
                except Exception as e:
                    print(f"Failed to load model file {model_path}: {e}")
                    continue

        # If no model file is found, try loading the default file
        default_model_path = os.path.join(path, "unet_heatmap.pth")
        print(
            f"No model file found, trying to load from default path: {default_model_path}")
        try:
            checkpoint = torch.load(default_model_path, map_location="cpu")
            # Check if it's a checkpoint containing multiple components
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Use the model state dict instead of the entire checkpoint
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Successfully loaded model_state_dict from checkpoint")
            else:
                # Try to load directly, assuming it's a simple model state dictionary
                self.model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Failed to load model file: {e}")
            print("Please ensure the model file exists and is named 'unet_heatmap.pth', 'model.pth', or 'heatmap_model.pth'")

        return self

    def predict(self, X):
        """
        Predicts landmark coordinates from a single grayscale PIL image
        using the same logic as `evaluate_model_mrd()`.
    
        Args:
            model (nn.Module): Trained PyTorch model
            pil_image (PIL.Image): Grayscale input image
            device (str): Device to run the model on
    
        Returns:
            coords (np.ndarray): [3, 2] array of predicted (x, y) keypoints
                                 in original image space
        """
        self.model.eval()

        # Step 1: Convert to grayscale and get original size
        pil_image = X.convert("L")
        W_orig, H_orig = pil_image.size

        # Step 2: Resize to 512x512 to match model training
        resized_image = pil_image.resize((512, 512))

        # Step 3: Prepare image for Albumentations
        image_np = np.array(resized_image)  # [H, W]
        if image_np.ndim == 2:
            image_np = image_np[:, :, None]  # [H, W, 1]

        transform = A.Compose([
            A.Normalize(mean=[0.0], std=[255.0]),
            ToTensorV2()
        ])
        augmented = transform(image=image_np)
        input_tensor = augmented["image"].unsqueeze(
            0).to(self.device)  # [1, 1, 512, 512]

        with torch.no_grad():
            _, coords = self.model(input_tensor)  # [1, 6]

        # scale back to original size
        return coords.reshape(-1, 6).cpu().numpy()*512

    def save(self, path="./"):
        '''
        Save model weights
        '''
        pass


def main():

    parser = argparse.ArgumentParser(
        description="Inference script for landmark detection using HRNet model")
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image (grayscale or RGB)')
    parser.add_argument('--model_dir', type=str, default='./',
                        help='Directory containing the model weights')

    args = parser.parse_args()

    # Load input image
    if not os.path.isfile(args.image_path):
        print(f"Image file not found: {args.image_path}")
        return

    image = Image.open(args.image_path)
    print(f"Loaded image: {args.image_path}, size: {image.size}")

    # Initialize and load model
    predictor = model().load(args.model_dir)

    # Run prediction
    coords = predictor.predict(image)

    # Print the results
    coords = coords.reshape(3, 2)  # Reshape to (3, 2) for readability
    for i, (x, y) in enumerate(coords, 1):
        print(f"Keypoint {i}: x = {x:.2f}, y = {y:.2f}")


if __name__ == "__main__":
    main()

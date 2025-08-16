import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import ast
import numpy as np
import cv2
import albumentations as A
from utils import soft_argmax, get_coordinates_from_heatmaps

class HeatmapLandmarkDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, train=True, num_views=None, heatmap_size=512, sigma=2.0):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.train = train
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.transform = transform
        self.num_views = num_views if hasattr(self, 'num_views') else 1

    def __len__(self):
        return len(self.data)

    def generate_heatmap(self, center_x, center_y, height, width):
        x = np.arange(0, width, 1, np.float32)
        y = np.arange(0, height, 1, np.float32)
        y = y[:, np.newaxis]
        heatmap = np.exp(-((x - center_x) ** 2 + (y - center_y)
                         ** 2) / (2 * self.sigma ** 2))
        return heatmap

    def __getitem__(self, index):
        row = self.data.iloc[index]
        filename = row['Filename']
        img_path = os.path.join(self.img_dir, filename)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Try parsing landmarks
        try:
            ps1 = ast.literal_eval(row["PS1"])
            ps2 = ast.literal_eval(row["PS2"])
            fh = ast.literal_eval(row["FH1"])
            keypoints = [list(ps1), list(ps2), list(fh)]
            if len(keypoints) != 3 or any(len(kp) != 2 for kp in keypoints):
                raise ValueError(f"Invalid keypoints at index {index}: {keypoints}")
        
        except Exception as e:
            raise ValueError(
                f"Error parsing keypoints at index {index}: {e}, row: {row}")
        

        # Ensure exactly 3 keypoints
        if not (isinstance(keypoints, list) and len(keypoints) == 3 and all(len(kp) == 2 for kp in keypoints)):
            raise ValueError(
                f"Invalid or incomplete keypoints at index {index}: {keypoints}")


        if self.transform:
            image, keypoints = self.apply_transform_with_keypoints(
                image=image, keypoints=keypoints, num_views=self.num_views)
            
            if not (isinstance(keypoints, list) and len(keypoints) == 3 and all(len(kp) == 2 for kp in keypoints)):
                raise ValueError(
                    f"Invalid or incomplete keypoints at index {index}: {keypoints}")


        # Generate heatmaps
        heatmaps = np.zeros(
            (3, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        img_height, img_width = image.shape[1], image.shape[2]
        scale_x = self.heatmap_size / img_width
        scale_y = self.heatmap_size / img_height

        for i, kp in enumerate(keypoints):
            if i >= len(keypoints) or len(keypoints[i]) != 2:
                raise IndexError(
                    f"Missing or invalid keypoint {i} at index {index}: {keypoints}")
            x = int(kp[0] * scale_x)
            y = int(kp[1] * scale_y)
            x = max(0, min(x, self.heatmap_size - 1))
            y = max(0, min(y, self.heatmap_size - 1))
            heatmaps[i] = self.generate_heatmap(
                x, y, self.heatmap_size, self.heatmap_size)

        heatmaps = torch.from_numpy(heatmaps)

        # Normalized landmark coordinates
        landmarks = [
            keypoints[0][0] / img_width, keypoints[0][1] / img_height,
            keypoints[1][0] / img_width, keypoints[1][1] / img_height,
            keypoints[2][0] / img_width, keypoints[2][1] / img_height
        ]

        landmarks = torch.tensor(landmarks, dtype=torch.float32)

        return image, heatmaps, landmarks

    

    def apply_transform_with_keypoints(self, image, keypoints, num_views=1):
        """
        Apply albumentations transform to image and keypoints, generating multiple views.
        
        Args:
            image (PIL.Image): Input image (grayscale)
            keypoints (list): List of (x, y) tuples
            num_views (int): Number of augmented views to generate

        Returns:
            If num_views == 1:
                image_tensor (torch.Tensor): Transformed image
                transformed_keypoints (list): Transformed (x, y) keypoints
            If num_views > 1:
                list of tuples [(image_tensor, transformed_keypoints), ...] for each view
        """
        # Convert PIL to numpy
        image_np = np.array(image)

        # Convert grayscale to 3D for albumentations if needed
        if image_np.ndim == 2:
            image_np = image_np[:, :, None]

        if num_views == 1:
            # Apply the transform
            transformed = self.transform(image=image_np, keypoints=keypoints)
            image_tensor = transformed["image"]
            transformed_keypoints = transformed["keypoints"]
            return image_tensor, transformed_keypoints
        else:
            # Generate multiple views
            views = []
            for _ in range(num_views):
                transformed = self.transform(image=image_np, keypoints=keypoints)
                views.append((transformed["image"], transformed["keypoints"]))
            return views


if __name__ == "__main__":

    # Test HeatmapLandmarkDataset
    image_dir = "Training\\Labeled cases"  # Windows path format
    csv_path = "Training\\Labeled cases\\label.csv"

    # Initialize dataset
    dataset = HeatmapLandmarkDataset(csv_file=csv_path, img_dir=image_dir, heatmap_size=512, sigma=10.0)
    print(f"Dataset initialized with {len(dataset)} samples.")

    print(f"Dataset size: {len(dataset)}")

    # Get first sample for testing
    idx = 0
    image, heatmaps, landmarks = dataset[idx]

    print(f"Image shape: {image.shape}")
    print(f"Heatmaps shape: {heatmaps.shape}")
    print(f"Landmarks (normalized): {landmarks}")

    # Convert heatmaps to batch format for get_coords_from_heatmap
    heatmaps_batch = heatmaps.unsqueeze(0)  # Add batch dimension [1, 3, H, W]
    coords = soft_argmax(heatmaps_batch)
    print(f"Regenerated coordinates (in heatmap space): {coords[0]}")

    # Original landmarks are in the format [x1, y1, x2, y2, x3, y3]
    # Reshape to [3, 2] and scale to heatmap size
    original_landmarks = landmarks.view(-1, 2)  # Reshape to [3, 2]
    scaled_landmarks = original_landmarks * dataset.heatmap_size
    print(f"Original landmarks (scaled to heatmap size): {scaled_landmarks}")

    # Compare
    all_passed = True
    for i in range(3):
        regen_x, regen_y = coords[0, i]
        orig_x, orig_y = scaled_landmarks[i]
        dist = torch.sqrt((regen_x.float() - orig_x)**2 + (regen_y.float() - orig_y)**2)
        print(f"Landmark {i}: Original ({orig_x:.1f}, {orig_y:.1f}), Regenerated ({regen_x}, {regen_y}), Distance: {dist:.2f}")
        
        # Check if the regenerated coordinate is close to the original
        if dist < 3:  # Within 3 pixels
            print(f"Landmark {i}: Coordinates regenerated successfully!")
        else:
            print(f"Landmark {i}: Regeneration differs by {dist:.2f} pixels")
            all_passed = False

    if all_passed:
        print("\nAll landmarks were regenerated successfully!")
    else:
        print("\nSome landmarks had significant differences in regeneration.")
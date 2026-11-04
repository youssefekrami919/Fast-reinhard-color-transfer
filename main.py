import os
import cv2
import numpy as np


# Reference / Source image path: the image providing the target color palette
source_image_path = r"source_image.jpg"

# Target image path: the image whose colors will be modified to match the source
target_image_path = r"target_image.jpg"


def get_mean_and_std(x):
    """Compute the mean and standard deviation for each color channel."""
    mean, std = cv2.meanStdDev(x)
    return mean.flatten(), std.flatten()


# 1. Read and prepare the source image
source_img = cv2.imread(source_image_path)
if source_img is None:
    raise FileNotFoundError(f"Failed to load source image from: {source_image_path}")

# Convert to CIELAB color space and extract statistics
source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
source_mean, source_std = get_mean_and_std(source_lab)

# 2. Read and prepare the target image
target_img = cv2.imread(target_image_path)
if target_img is None:
    raise FileNotFoundError(f"Failed to load target image from: {target_image_path}")

# Convert to CIELAB color space using float32 to prevent arithmetic overflow/underflow
target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)
target_mean, target_std = get_mean_and_std(target_lab)

# 3. Apply Reinhard color transfer algorithm
# Compute scale ratio with an epsilon safeguard against division by zero
scale = source_std / (target_std + 1e-5)

# Match statistical color distribution via vectorized array operations
result_lab = ((target_lab - target_mean) * scale) + source_mean

# Clip values to the valid [0, 255] range and cast back to uint8
result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)

# Convert image back to standard BGR color space
output_img = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

# 4. Save output in the exact same directory as the target image
target_dir, target_filename = os.path.split(target_image_path)
output_path = os.path.join(target_dir, f"modified_{target_filename}")

cv2.imwrite(output_path, output_img)
print(f"Image successfully processed and saved to: {output_path}")
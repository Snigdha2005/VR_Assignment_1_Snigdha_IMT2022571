import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from skimage import measure, color, morphology

output_dir = 'output_images'

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create output directory if it doesn't exist
os.makedirs(output_dir)

# Load the image
image_path = 'input_images/scatter.jpg'  # Path to the uploaded image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray, (15, 15), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred, 50, 220)

# Find contours from the edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def draw_contours(image, contours):
    """ Draw contours around detected coins """
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 3)  # Draw green contours
    return output

# Draw contours on the original image
detected_coins = draw_contours(image, contours)

# Save detected coins image
cv2.imwrite(os.path.join(output_dir, 'detected_coins.jpg'), detected_coins)
cv2.imwrite(os.path.join(output_dir, 'edges.jpg'), edges)

# Segmentation of each coin
segmented_coins = []
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    coin = image[y:y+h, x:x+w]  # Crop each detected coin
    segmented_coins.append(coin)
    cv2.imwrite(os.path.join(output_dir, f'coin_{i+1}.jpg'), coin)

# Count the total number of coins
total_coins = len(contours)

# Display results
print(f"[Edge Detection + Contour] Total number of coins detected: {total_coins}")

# Display results
plt.figure(figsize=(18, 7))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Canny Edges')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Detected Coins (Contours)')
plt.imshow(cv2.cvtColor(detected_coins, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

# Load image
image_path = "input_images/scatter.jpg"  # Update this with your image path
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Apply Gaussian Blur to reduce noise (increased from (1,1) to (5,5))
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 2: Apply Otsu’s thresholding to segment foreground and background
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 3: Morphological operations (kernel size increased from (1,1) to (3,3))
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 4: Label connected components (changed connectivity from 1 → 2)
labels = measure.label(cleaned, connectivity=2)

# Step 5: Reduce `min_size` from 500 → 200 to keep smaller coins
labels = morphology.remove_small_objects(labels, min_size=201)

# Step 6: Convert labels into a colored segmentation mask
colored_labels = color.label2rgb(labels, bg_label=0)

# Step 7: Count total number of detected coins
num_coins = len(np.unique(labels)) - 1  # Exclude background

print(f"[Segmentation] Total Coins Detected: {num_coins}")

cv2.imwrite(os.path.join(output_dir, 'region_based_segmented_coins.jpg'), colored_labels)

# Display results
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Segmented Coins (Region-Based)")
plt.imshow(colored_labels)
plt.axis("off")

plt.show()

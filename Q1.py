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
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred, 0, 234)

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
plt.imshow(edges, cmap='nipy_spectral')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Detected Coins (Contours)')
plt.imshow(cv2.cvtColor(detected_coins, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 2: Apply Otsu’s thresholding (Ensure foreground is white)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 4: Morphological operations to remove noise
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 5: Label connected components
labels = measure.label(cleaned, connectivity=2)

# Step 6: Remove small objects (Set min_size based on coin size)
labels = morphology.remove_small_objects(labels, min_size=2000)  # Adjust if needed

# Step 7: Convert labeled regions into a colored mask
colored_labels = color.label2rgb(labels, bg_label=0)

# Step 8: Count total number of detected coins
num_coins = len(np.unique(labels)) - 1  # Exclude background

print(f"[Segmentation] Total Coins Detected: {num_coins}")

# Step 9: Display the segmented coins properly
plt.figure(figsize=(6, 6))
plt.imshow(colored_labels)
plt.title("Segmented Coins (Corrected)")
plt.axis("off")
plt.show()

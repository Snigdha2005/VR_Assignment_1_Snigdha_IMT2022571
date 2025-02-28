import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Define paths
input_folder = "input_images/"
output_folder = "output_images/"

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def detect_and_save_keypoints(image, filename):
    """ Detects keypoints in an image and saves the result. """
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    # Draw keypoints on the image
    keypoint_image = cv2.drawKeypoints(image, keypoints, None, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Save keypoint image
    keypoint_path = os.path.join(output_folder, filename)
    cv2.imwrite(keypoint_path, keypoint_image)
    print(f"Keypoints saved: {keypoint_path}")

def stitch_images(images):
    """ Stitches overlapping images into a single panorama. """
    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    status, stitched_image = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print("Panorama created successfully!")
        return stitched_image
    else:
        print("Error during stitching:", status)
        return None

# Load images from folder (ensure filenames are in order)
image_paths = sorted(glob.glob(os.path.join(input_folder, "*_panorama.jpg")))  # Change if needed
images = [cv2.imread(img) for img in image_paths]

# Process each image to detect and save keypoints
for i, img in enumerate(images):
    detect_and_save_keypoints(img, f"keypoints_{i+1}.jpg")

# Create panorama
panorama = stitch_images(images)

# Show and save results
if panorama is not None:
    panorama_path = os.path.join(output_folder, "stitched_panorama.jpg")
    cv2.imwrite(panorama_path, panorama)
    print(f"Panorama saved: {panorama_path}")

    # Display stitched panorama
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Stitched Panorama")
    plt.show()

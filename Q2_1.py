import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Create output directory
output_dir = "./output_images"
os.makedirs(output_dir, exist_ok=True)

# Load images from "input_images" folder with filenames containing "_panorama"
image_paths = sorted(glob.glob("./input_images/*_panorama.jpg"))

# Read images
images = [cv2.imread(img) for img in image_paths]
if len(images) < 2:
    print("At least two images are required for stitching!")
    exit()

# Resize images for consistency
def resize_images(images, width=800):
    return [cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1]))) for img in images]

images = resize_images(images)

# Detect keypoints and match features using SIFT
def detect_and_match_keypoints(img1, img2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Draw keypoints on the images and save them

    img1_with_keypoints = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_with_keypoints = cv2.drawKeypoints(img2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
    keypoints1_path = os.path.join(output_dir, f"1_keypoints.jpg")
    keypoints2_path = os.path.join(output_dir, f"2_keypoints.jpg")
    cv2.imwrite(keypoints1_path, img1_with_keypoints)
    cv2.imwrite(keypoints2_path, img2_with_keypoints)
    print(f"Saved keypoints")

    # Use Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort based on distance

    return keypoints1, keypoints2, matches

# Warp images using homography
def warp_images(img1, img2):
    keypoints1, keypoints2, matches = detect_and_match_keypoints(img1, img2)

    # Extract matching points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography (mapping img2 onto img1)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Warp second image onto first
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create a larger canvas for stitching
    panorama_width = w1 + w2  
    panorama_height = max(h1, h2)  

    panorama = cv2.warpPerspective(img2, H, (panorama_width, panorama_height))
    panorama[0:h1, 0:w1] = img1  # Overlay the first image

    return panorama

# Trim black areas from the final panorama
def trim_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    trimmed = image[y:y+h, x:x+w]
    return trimmed

# Stitch multiple images sequentially
def stitch_panorama(images):
    panorama = images[0]  # Start with the first image

    for i in range(1, len(images)):
        print(f"Stitching image {i+1} of {len(images)}...")
        panorama = warp_images(panorama, images[i])

    return trim_black_borders(panorama)

# Generate the panorama
panorama = stitch_panorama(images)

# Save and display results
output_path = os.path.join(output_dir, "stitched_panorama.jpg")
cv2.imwrite(output_path, panorama)

# Show stitched panorama
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Stitched Panorama")
plt.show()

print(f"Panorama saved at {output_path}")

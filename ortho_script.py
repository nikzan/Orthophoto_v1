import cv2
import numpy as np
import glob
import uuid
from tqdm import tqdm


def detect_and_describe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def match_descriptors(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def find_homography(kp1, kp2, matches):
    if len(matches) < 4:
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H


def combine_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)

    all_corners = np.concatenate((warped_corners,
                                  np.float32([[0, 0], [w1, 0], [0, h1], [w1, h1]]).reshape(-1, 1, 2)),
                                 axis=0)

    [x_min, y_min] = np.int32(np.min(all_corners, axis=(0, 1)) - 1)
    [x_max, y_max] = np.int32(np.max(all_corners, axis=(0, 1)) + 1)

    translation = np.array([[1, 0, -x_min],
                            [0, 1, -y_min],
                            [0, 0, 1]])

    warped_img2 = cv2.warpPerspective(img2, translation.dot(H),
                                      (x_max - x_min, y_max - y_min))

    panorama = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    panorama[-y_min:h1 - y_min, -x_min:w1 - x_min] = img1

    mask = warped_img2 > 0
    panorama[mask] = warped_img2[mask]

    return panorama


def stitch_images(images):
    if len(images) < 1:
        return None

    result = images[0]
    for i in tqdm(range(1, len(images)), desc="Stitching images"):
        kp1, des1 = detect_and_describe(result)
        kp2, des2 = detect_and_describe(images[i])

        matches = match_descriptors(des1, des2)
        if len(matches) < 4:
            print(f"Not enough matches for image {i + 1}")
            continue

        H = find_homography(kp1, kp2, matches)
        if H is None:
            print(f"Homography failed for image {i + 1}")
            continue

        H_inv = np.linalg.inv(H)
        result = combine_images(result, images[i], H_inv)

    return result


if __name__ == "__main__":
    images = []
    for img_path in sorted(glob.glob('input0/*.JPG')):
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Error loading: {img_path}")

    if len(images) < 2:
        print("Need at least 2 images")
    else:
        # Generate unique filename using UUID
        unique_filename = f'panorama_result_{uuid.uuid4().hex}.jpg'
        panorama = stitch_images(images)
        if panorama is not None:
            cv2.imwrite(unique_filename, panorama)
            print(f"Panorama saved as {unique_filename}")
        else:
            print("Panorama creation failed")
import cv2
import re
import numpy as np
import glob
from collections import defaultdict
import open3d as o3d

def load_images(paths):
    """Load images from given paths and validate they were loaded correctly."""
    images = [cv2.imread(p) for p in paths]
    for i, img in enumerate(images):
        if img is None:
            raise FileNotFoundError(f"Image {paths[i]} could not be loaded.")
    return images

def read_calibration_matrix(path):
    """Read camera calibration matrix from a text file."""
    with open(path, 'r') as f:
        vals = re.findall(r"[-+]?\d*\.\d+|\d+", f.read())
        K = np.array(vals[:9], dtype=np.float32).reshape(3, 3)
    return K

def load_data(image_dir, calibration_file):
    """Load images and calibration data from directory."""
    image_paths = sorted(glob.glob(f"{image_dir}\\*.jpg"))
    if len(image_paths) < 2:
        raise RuntimeError("At least two images are required.")
    images = load_images(image_paths)
    K = read_calibration_matrix(calibration_file)
    return images, K, image_paths

def match_all_images(images):
    """Extract features and match across all images using SIFT."""
    sift = cv2.SIFT_create(nfeatures=10000)
    keypoints_list, descriptors_list = [], []
    
    # Extract features from each image
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        keypoints_list.append(kp)
        descriptors_list.append(des)
    
    # Match features between all image pairs
    matcher = cv2.BFMatcher()
    matches_dict = defaultdict(list)
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            matches = matcher.knnMatch(descriptors_list[i], descriptors_list[j], k=2)
            # Apply ratio test to filter good matches
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
            if len(good_matches) >= 8:  # Minimum matches for fundamental matrix
                matches_dict[(i, j)] = good_matches
    return keypoints_list, matches_dict

def triangulate_two_views(kp1, kp2, matches, K, R1, t1, R2, t2):
    """Triangulate 3D points from two views."""
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    P1 = K @ np.hstack((R1, t1))  # Projection matrix 1
    P2 = K @ np.hstack((R2, t2))  # Projection matrix 2
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    return (pts4d[:3] / pts4d[3]).T  # Convert from homogeneous to 3D

def enhance_color(rgb, contrast=1.4, saturation=1.3):
    """Enhance colors with contrast and saturation adjustments."""
    mean = np.mean(rgb)
    rgb = mean + contrast * (rgb - mean)
    rgb = np.clip(rgb, 0, 1)

    import colorsys
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = min(s * saturation, 1.0)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return np.clip([r, g, b], 0, 1)

def get_point_colors(image1, image2, kp1, kp2, matches, mask=None):
    """Extract colors for matched points from both images."""
    colors = []
    for idx, m in enumerate(matches):
        if mask is not None and not mask[idx]:
            continue
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
        x2, y2 = int(round(pt2[0])), int(round(pt2[1]))
        bgr1 = image1[y1, x1]
        bgr2 = image2[y2, x2]
        rgb = ((bgr1[::-1].astype(float) + bgr2[::-1].astype(float)) / 2) / 255.0
        enhanced = enhance_color(rgb)
        colors.append(enhanced)
    return np.array(colors)

def merge_point_clouds(pcd1, pcd2, scale_factor, delta_x, delta_y, delta_z):
    """Merge two point clouds with scaling and translation."""
    if pcd1 is None or pcd2 is None:
        print("[ERROR] One of the point clouds is empty.")
        return

    pcd2.scale(scale_factor, center=pcd2.get_center())  # Apply scaling to the second point cloud

    T = np.eye(4)
    T[:3, 3] = [delta_x, delta_y, delta_z]  # Translation matrix
    pcd2.transform(T)  # Apply translation

    combined = pcd1 + pcd2  # Combine point clouds
    o3d.io.write_point_cloud("combined_output.ply", combined)  # Save combined cloud
    o3d.visualization.draw_geometries([combined])  # Visualize the result
    return combined

def load_and_reconstruct(image_dir, calibration_file):
    # Load images and camera calibration matrix
    images, K, _ = load_data(image_dir, calibration_file)
    keypoints_list, matches_dict = match_all_images(images)

    poses = {}
    all_points = []
    colors_all = []
    # Take the first pair of images from the matches
    idx1, idx2 = list(matches_dict.keys())[0]
    kp1, kp2 = keypoints_list[idx1], keypoints_list[idx2]
    matches = matches_dict[(idx1, idx2)]
    # Extract 2D points for triangulation from the keypoints in the matched pair
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    # Find the essential matrix E using the 2D points, and recover the relative pose
    E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    # Assign initial poses: The first image has the identity matrix (no rotation, no translation)
    poses[idx1] = (np.eye(3), np.zeros((3, 1)))
    poses[idx2] = (R, t)
    # Triangulate 3D points for the initial pair of images
    pts3d = triangulate_two_views(kp1, kp2, matches, K, poses[idx1][0], poses[idx1][1], poses[idx2][0], poses[idx2][1])
    colors = get_point_colors(images[idx1], images[idx2], kp1, kp2, matches)
    colors_all.append(colors)

    all_points.append(pts3d)
    # Loop through all matches and reconstruct 3D points for all other images
    for (i, j), matches in matches_dict.items():
        if i in poses and j not in poses:
            known, new = i, j
        elif j in poses and i not in poses:
            known, new = j, i
        else:
            continue
        # Extract keypoints for known and new images
        kp_known = keypoints_list[known]
        kp_new = keypoints_list[new]
        R_known, t_known = poses[known]

        pts3d_tmp = triangulate_two_views(kp_known, kp_new, matches, K, R_known, t_known, np.eye(3), np.zeros((3, 1)))
        pts2d = np.float32([kp_new[m.trainIdx].pt if known == i else kp_new[m.queryIdx].pt for m in matches])
        #Removing outliers
        mask = np.linalg.norm(pts3d_tmp, axis=1) < 1000
        pts3d_tmp = pts3d_tmp[mask]
        pts2d = pts2d[mask]

        if len(pts3d_tmp) < 6:
            continue
	#PnP
        success, rvec, tvec = cv2.solvePnP(pts3d_tmp, pts2d, K, None)
        if not success:
            continue

        R_new, _ = cv2.Rodrigues(rvec)
        poses[new] = (R_new, tvec)
        pts3d_new = triangulate_two_views(kp_known, kp_new, matches, K, R_known, t_known, R_new, tvec)
        colors = get_point_colors(images[known], images[new], kp_known, kp_new, matches)
        colors_all.append(colors)

        all_points.append(pts3d_new)
	#Point cloud creation from 3D points
    if all_points:
        cloud = np.vstack(all_points)
        colors = np.vstack(colors_all)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd
    else:
        return None

if __name__ == "__main__":
    # === 1. Front: ETF2 + ETF3 ===
    # Reconstruct front view from two image sets
    pcd1 = load_and_reconstruct("etf2", "K.txt")
    pcd2 = load_and_reconstruct("etf3", "K.txt")
    front_scene = merge_point_clouds(pcd1, pcd2, 1.2, 65, 2, 2)
    
    # === 2. Side view (side1) - rotate 90° around Y axis ===
    pcd3 = load_and_reconstruct("strana1", "K.txt")
    R = pcd3.get_rotation_matrix_from_axis_angle([0, np.pi / 2, 0])
    pcd3.rotate(R, center=pcd3.get_center())
    final_scene = merge_point_clouds(front_scene, pcd3, 1.0, 111, -1, -3)
    
    # Additional rotation adjustment
    rad_x = np.deg2rad(0)
    R_z = pcd3.get_rotation_matrix_from_axis_angle([rad_x, 0, 0])
    pcd3.rotate(R_z, center=pcd3.get_center())
    final_scene2 = merge_point_clouds(final_scene, pcd3, 1.0, -129, -1, 0)

    # === 3. Back and side2 → scene3 ===
    pcd4 = load_and_reconstruct("zadnja", "K.txt")
    pcd5 = load_and_reconstruct("strana2", "K.txt")

    bbox1 = pcd4.get_max_bound() - pcd4.get_min_bound()
    bbox2 = pcd5.get_max_bound() - pcd5.get_min_bound()
    scale_ratio = (bbox1[0] / bbox2[0]) * 0.65
    pcd5.scale(scale_ratio, center=pcd5.get_center())

    scene3 = merge_point_clouds(pcd4, pcd5, 1.0, -15, 14, -33)

    # === 4. Scale scene3 to match height (Y) of final_scene2 ===
    bbox_final = final_scene2.get_max_bound() - final_scene2.get_min_bound()
    bbox_3 = scene3.get_max_bound() - scene3.get_min_bound()
    scale_scene3 = bbox_final[1] / bbox_3[1]  # Scaling based on height (Y axis)
    scene3.scale(scale_scene3, center=scene3.get_center())

    # === 5. Extend scene3 along X axis (length) ===
    points = np.asarray(scene3.points)
    center = scene3.get_center()
    points = (points - center) * np.array([1.1, 1.0, 1.0]) + center
    scene3.points = o3d.utility.Vector3dVector(points)

    # === 6. Merge final_scene2 + scene3 ===
    final_merged = merge_point_clouds(final_scene2, scene3, 1.0, 72, 13, -45)
    
    # === Post-processing ===
    # 1. Remove noise and outliers
    final_clean, ind = final_merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 2. Rotate 180° around Y axis for better viewing
    R_flip = final_clean.get_rotation_matrix_from_axis_angle([0, np.pi, 0])
    final_clean.rotate(R_flip, center=final_clean.get_center())

    # Visualize final result
    o3d.visualization.draw_geometries([final_clean])

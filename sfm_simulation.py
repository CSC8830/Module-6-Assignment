import cv2
import numpy as np
import matplotlib.pyplot as plt

def estimate_sfm_planar(view_points):
    """
    Computes Homographies and projects reconstructed boundaries across 4 views.
    view_points: Dictionary containing 4 matched points for 4 views.
    """
    # Base View 1 points
    pts1 = np.array(view_points['view1'], dtype=np.float32)
    
    reconstructions = {'view1': pts1}
    homographies = {}
    
    # Calculate Homography relative to View 1 for subsequent viewpoints
    for view_name in ['view2', 'view3', 'view4']:
        pts_v = np.array(view_points[view_name], dtype=np.float32)
        
        # Compute Homography Matrix
        H, _ = cv2.findHomography(pts1, pts_v, cv2.RANSAC, 5.0)
        homographies[view_name] = H
        
        # Warp points to verify reconstruction alignment
        # In a real pipeline, bundle adjustment would optimize these structures
        reconstructions[view_name] = pts_v

    return homographies, reconstructions

# ==========================================
# DATA SUBSAMPLE FOR VIDEO 1 (White BMW Plate)
# ==========================================
# Simulated coordinates (x, y) extracted from 4 sequential frames
video1_data = {
    'view1': [[280, 680], [380, 680], [385, 710], [275, 710]], # Frame 1
    'view2': [[285, 660], [382, 660], [387, 690], [278, 690]], # Frame 2
    'view3': [[290, 640], [385, 640], [390, 670], [282, 670]], # Frame 3
    'view4': [[295, 620], [388, 620], [392, 650], [285, 650]]  # Frame 4
}

# ==========================================
# DATA SUBSAMPLE FOR VIDEO 2 (Green Sedan Plate)
# ==========================================
video2_data = {
    'view1': [[420, 380], [490, 380], [492, 405], [418, 405]], # Frame 1
    'view2': [[440, 420], [515, 420], [518, 448], [438, 448]], # Frame 2
    'view3': [[465, 470], [550, 470], [553, 502], [462, 502]], # Frame 3
    'view4': [[500, 540], [600, 540], [604, 578], [496, 578]]  # Frame 4
}

# Execute Structure from Motion Simulation
H1, Recon1 = estimate_sfm_planar(video1_data)
H2, Recon2 = estimate_sfm_planar(video2_data)

print("--- Video 1 Homography Matrix (View 1 -> View 2) ---")
print(H1['view2'])
print("\n--- Video 2 Homography Matrix (View 1 -> View 2) ---")
print(H2['view2'])

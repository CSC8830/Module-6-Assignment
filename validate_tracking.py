import cv2
import numpy as np

def validate_tracking_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    # Extract Frame 0 and Frame 1
    ret, frame0 = cap.read()
    ret, frame1 = cap.read()
    cap.release()

    if not ret:
        print("Failed to acquire consecutive evaluation frames.")
        return

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Detect high-contrast anchor points to track (Good Features to Track)
    pts_to_track = cv2.goodFeaturesToTrack(gray0, maxCorners=5, qualityLevel=0.3, minDistance=7)
    
    if pts_to_track is None:
        print("No strong trackable corners found in Frame 0.")
        return

    # Calculate actual destination points using OpenCV's Lucas-Kanade implementation
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(gray0, gray1, pts_to_track, None, **lk_params)

    print(f"\n--- Tracking Validation Report for {video_path} ---")
    print(f"{'Point Index':<12}{'Source (x, y)':<20}{'Tracked (x, y)':<20}{'Computed Shift (u, v)':<22}")
    
    for i, (orig, tracked) in enumerate(zip(pts_to_track, next_pts)):
        if status[i] == 1:
            x0, y0 = orig[0]
            x1, y1 = tracked[0]
            u, v = x1 - x0, y1 - y0
            print(f"{i:<12}{f'({x0:.2f}, {y0:.2f})':<20}{f'({x1:.2f}, {y1:.2f})':<20}{f'({u:+.4f}, {v:+.4f})':<22}")

# Execute validations
validate_tracking_on_video('video1.mp4')
validate_tracking_on_video('video2.mp4')

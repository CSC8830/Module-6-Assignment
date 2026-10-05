import cv2
import numpy as np

def compute_and_save_optical_flow(video_path, output_path):
    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Video has no frames.")
        return
        
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    # Create an HSV mask to visualize the flow direction and magnitude
    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255  # Set saturation to maximum

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate Dense Optical Flow using Gunnar Farneback's algorithm
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Compute magnitude and angle of the 2D flow vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Map flow directions to HSV Hue, and magnitudes to HSV Value
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert HSV back to BGR for visualization
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Write frame to output video file
        out.write(flow_bgr)
        
        # Progress state transition
        prev_gray = next_gray

    cap.release()
    out.release()
    print(f"Optical flow video successfully saved to {output_path}")

# Example usages for your files:
# compute_and_save_optical_flow('video1.mp4', 'output_flow_video1.mp4')
# compute_and_save_optical_flow('video2.mp4', 'output_flow_video2.mp4')

if __name__ == "__main__":
    compute_and_save_optical_flow('video1.mp4', 'output_flow_video1.mp4')
    compute_and_save_optical_flow('video2.mp4', 'output_flow_video2.mp4')

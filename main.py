import cv2
import numpy as np

from util import get_parking_spots_bboxes, empty_or_not

# Function to calculate the absolute difference in mean intensity between two images
def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# File paths for the mask and video
mask_path = './video_and_image/mask.png'
video_path = './video_and_image/parking.mp4'

# Output video file path
output_video_path = './video_and_image/parking_processed.mp4'

# Read the mask image and the video
mask = cv2.imread(mask_path, 0)
cap = cv2.VideoCapture(video_path)

# Get video properties (width, height, frames per second)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create a VideoWriter object to save the processed video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Use connected components to identify parking spots in the mask
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)
spots_status = [None for j in spots]
diffs = [None for j in spots]

# Variable to store the previous frame for comparison
previous_frame = None

# Frame number counter
frame_nmr = 0

# Read frames from the video
ret = True
step = 30  # Process every 'step' frames
while ret:
    ret, frame = cap.read()

    # Check if it's time to process the frame
    if frame_nmr % step == 0 and previous_frame is not None:
        # Calculate the difference for each parking spot
        for spot_index, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_index] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        # Print differences sorted in descending order
        print([diffs[j] for j in np.argsort(diffs)][::-1])

    # Check if it's time to process the frame
    if frame_nmr % step == 0:
        # Determine spots to process based on differences
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]

        # Process selected spots
        for spot_index in arr_:
            spot = spots[spot_index]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_index] = spot_status

    # Check if it's time to process the frame
    if frame_nmr % step == 0:
        # Save the current frame for the next iteration
        previous_frame = frame.copy()

    # Draw rectangles and display results on the frame
    for spot_index, spot in enumerate(spots):
        spot_status = spots_status[spot_index]
        x1, y1, w, h = spots[spot_index]

        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)  # Green for empty
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)  # Red for occupied

    # Display the available spots information on the frame
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))),
                (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Write the frame to the output video
    out.write(frame)

    # Show the frame in a window
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Increment frame number
    frame_nmr += 1

# Release video capture, video writer, and close windows
cap.release()
out.release()
cv2.destroyAllWindows()

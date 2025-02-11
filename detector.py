import cv2
import numpy as np

# Video file path (update this to your correct path)
video_path = 'D:/4TH SEM/Python Project/Project.py/Video1.mp4'

# Camera
cap = cv2.VideoCapture(video_path)

# Check if the video file is loaded correctly
if not cap.isOpened():
    print(f"Error: Couldn't open video file at {video_path}. Please check the file path.")
    exit()

min_width_rect = 80  # Min width rectangle
min_height_rect = 80  # Min height rectangle

count_line_position = 550
# Initialize Subtractor
algorithm = cv2.createBackgroundSubtractorMOG2()

# Get the FPS of the video to adjust the wait time for 0.5x speed
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate delay for 0.5x speed (double the original delay)
delay = int(1000 / fps / 2)  # In milliseconds

while True:
    ret, frame1 = cap.read()

    # Check if frame is read properly
    if not ret:
        print("Error: Couldn't read a frame from the video.")
        break

    # Preprocess the frame: Convert to grayscale and blur to reduce noise
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 5)

    # Apply background subtraction
    img_sub = algorithm.apply(blur)

    # Dilate to fill gaps and improve contour detection
    dilat = cv2.dilate(img_sub, np.ones((5, 5), np.uint8))

    # Apply morphological transformations to close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)

    # Find contours
    countershape, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Drawing a line to count objects
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (155, 1270), 3)

    # Loop through each contour found and draw bounding boxes
    for (i, c) in enumerate(countershape):
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Validate contours by checking the size (width and height)
        if w >= min_width_rect and h >= min_height_rect:
            # Draw bounding box around detected vehicle
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # You can also add a label or other information here
            # cv2.putText(frame1, "Vehicle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the original video with detections
    cv2.imshow('Video Original', frame1)

    # Exit when 'Enter' key is pressed
    if cv2.waitKey(delay) == 13:  # Delay added here to slow down the video
        break

# Clean up
cv2.destroyAllWindows()
cap.release()

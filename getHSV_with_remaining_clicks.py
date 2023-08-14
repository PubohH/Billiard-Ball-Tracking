import cv2
import numpy as np

# Global variables
selected_frame = None
hsvList = []
remaining_clicks = 10  # Initial number of remaining clicks

# Initialize the webcam
cap = cv2.VideoCapture('3CTest1.mp4')

# Define a callback function to get the HSV color coordinate of the pixel
def get_color(event, x, y, flags, param):
    global selected_frame, hsvList, remaining_clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if selected_frame is not None:
            hsv = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2HSV)
            hsvList.append(hsv[y, x])
            remaining_clicks -= 1
            if remaining_clicks == 0:
                calculate_hsv_bounds()
            else:
                update_instructions()

# Update the instructions text
def update_instructions():
    global selected_frame, remaining_clicks
    frame_with_text = selected_frame.copy()
    instructions = f"Click {remaining_clicks} more times evenly on the ball you wish to track"
    cv2.putText(frame_with_text, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('frame', frame_with_text)

# Read and store the 10th frame
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    selected_frame = frame

# Set a mouse callback function to get the HSV color coordinate of the pixel
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', get_color)

# Shrink the selected frame to 0.8 by 0.8
selected_frame = cv2.resize(selected_frame, (0, 0), fx=0.8, fy=0.8)

# Update instructions before entering the loop
update_instructions()

# Show the selected frame
cv2.imshow('frame', selected_frame)

# Wait for user input and exit on 'q' key press
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

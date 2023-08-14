import cv2
import numpy as np
from collections import deque
# run the code, then first frame of your video will be displayed,
# click 10 times evenly across the ball you want to track, then press 'q', the tracked video will display and output to a mp4 file

# Global variables
selected_frame = None
hsvList = []
sens = 10 #tolerance of HSV coordinates, change this (20, 30, 40, etc) to achieve best tracking.
lower_color = None
upper_color = None
trajectory = deque(maxlen=10000)
videoname = 'poolTest1' #input video clip name
cap = cv2.VideoCapture(videoname + '.mp4') #可以直接输入视频名称，或者输入args.image后用cmd运行


# Define a callback function to get the HSV color coordinate of the pixel
def get_color(event, x, y, flags, param):
    global selected_frame, hsvList
    if event == cv2.EVENT_LBUTTONDOWN:
        if selected_frame is not None:
            hsv = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2HSV)
            hsvList.append(hsv[y, x])
            if len(hsvList) == 10:
                calculate_hsv_bounds()

# Calculate lower and upper HSV bounds based on stored HSV coordinates
def calculate_hsv_bounds():
    global hsvList, lower_color, upper_color
    hsvArray = np.array(hsvList)
    lower_bound = np.min(hsvArray, axis=0)
    upper_bound = np.max(hsvArray, axis=0)
    print("Lower HSV bound:", lower_bound)
    print("Upper HSV bound:", upper_bound)
    lower_color = lower_bound - sens
    upper_color = upper_bound + sens
    hsvList.clear()
# Initialize the webcam

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    selected_frame = frame

# Set a mouse callback function to get the HSV color coordinate of the pixel
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', get_color)

# Show the selected frame for HSV color selection
selected_frame = cv2.resize(selected_frame, (0, 0), fx=0.8, fy=0.8)
cv2.imshow('frame', selected_frame)

# Wait for user input and exit on 'q' key press
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  close all windows

cv2.destroyAllWindows()

# Continue with the second code using calculated lower_color and upper_color
# ... (paste the second code from the previous response here)



# Continue with the second code using calculated lower_color and upper_color
# ... (paste the second code from the previous response here)



# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.

result = cv2.VideoWriter(videoname + ' with trajectory.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         60, size)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the frame to get only the color pixels
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # Convert the mask to a binary image
    binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    # Find the contours of the color objects in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    if max_contour is not None:
        # Find the center and radius of the circle that encloses the contour
        (x, y), radius = cv2.minEnclosingCircle(max_contour)

        # Only consider the circle if it has a radius greater than 20 pixels
        if radius > 6:
            # Draw the center of the circle
            center = (int(x), int(y))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            # Draw the boundary of the circle
            #to be filled in here

            # Draw the trajectory of the circle for the past 2 seconds
            trajectory.appendleft(center)
            for i in range(1, len(trajectory)):
                if trajectory[i - 1] is None or trajectory[i] is None:
                    continue
                cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 255), 2)

    # Show the frame
    result.write(frame)
    frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)  # 将视频缩放，高=fy，宽=fx
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
# cap.release()
cv2.destroyAllWindows()
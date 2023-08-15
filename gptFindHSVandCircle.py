import cv2
import numpy as np
from collections import deque
import argparse
# run the code, then first frame of your video will be displayed,
# click 10 times evenly across the ball you want to track, then press 'q', the tracked video will display and output to a mp4 file

parser = argparse.ArgumentParser(description="A simple program that demonstrates argparse")
parser.add_argument('--input', '-i', type=str, default='poolTest2', help='Input file path')
parser.add_argument('--write-result', '-w', type=bool, default=False, help='Write result of tracking in [input] with trajectory.mp4')
args = parser.parse_args()

# Global variables
write_result = False
selected_frame = None
hsvList = []
sens = 40 #tolerance of HSV coordinates, change this (20, 30, 40, etc) to achieve best tracking.
lower_color = None
upper_color = None
trajectory = deque(maxlen=10000)
videoname = args.input #input video clip name
print('Initialzed Variables...')
cap = cv2.VideoCapture(videoname) #可以直接输入视频名称，或者输入args.image后用cmd运行
print('Captured Video...')

# Define a callback function to get the HSV color coordinate of the pixel
def get_color(event, x, y, flags, param):
    global selected_frame, hsvList
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2HSV)
        hsvList.append(hsv[y, x])
        print(hsvList)
        if len(hsvList) > 10:
            calculate_hsv_bounds()

# Calculate lower and upper HSV bounds based on stored HSV coordinates
def calculate_hsv_bounds():
    global hsvList, lower_color, upper_color
    hsvArray = hsvList
    lower_bound = np.min(hsvArray, axis=0)
    upper_bound = np.max(hsvArray, axis=0)
    print("Lower HSV bound:", lower_bound)
    print("Upper HSV bound:", upper_bound)
    lower_color = lower_bound #- sens
    upper_color = upper_bound #+ sens
    cv2.destroyAllWindows()
# Initialize the webcam

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
print('Click on the screen to get HSV readings')

ret, frame = cap.read()
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

selected_frame = frame

# Show the selected frame for HSV color selection
selected_frame = cv2.resize(selected_frame, size, fx=0.4, fy=0.4)
cv2.imshow('frame', selected_frame)

# Set a mouse callback function to get the HSV color coordinate of the pixel
cv2.setMouseCallback('frame', get_color)

# Wait for user input and exit on 'q' key press
if cv2.waitKey() & 0xFF == ord('q'):
    cv2.destroyAllWindows()

#  close all windows



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

if write_result:
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
    #TODO:Figure out the erode and dilate mechanisms
    '''
    This part is the key to getting precise contour of the object
    '''
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # Convert the mask to a binary image
    binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('binary', mask)
    cv2.waitKey()
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
        ellipse = cv2.fitEllipse(max_contour)
        center, (major_axis, minor_axis), angle = ellipse

        # Draw the fitted ellipse
        cv2.ellipse(frame, ellipse, (0, 0, 255), 2)


        # Draw the center of the fitted ellipse
        cv2.circle(frame, (int(center[0]), int(center[1])), 4, (255, 0, 0), -8)
        
        # Draw the trajectory of the circle for the past 2 seconds
        trajectory.appendleft((int(center[0]), int(center[1])))
        for i in range(1, len(trajectory)):
            if trajectory[i - 1] is None or trajectory[i] is None:
                continue
            cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 255), 2)

    # Show the frame
    if write_result:
        result.write(frame)
    frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)  # 将视频缩放，高=fy，宽=fx
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
# cap.release()
cv2.destroyAllWindows()
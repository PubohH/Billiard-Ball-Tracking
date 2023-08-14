import cv2
import numpy as np
from collections import deque

# Initialize the webcam
videoname = '3CTest1' #input video clip name
cap = cv2.VideoCapture(videoname + '.mp4') #可以直接输入视频名称，或者输入args.image后用cmd运行



# lower_color = np.array([40,130,0])
# upper_color = np.array([150,200,10])
#
#
sens = 20 #change this to adjust lower/upper hsv color tolerance
lower_color= np.array([20, 30, 200]) - sens
upper_color = np.array([40, 60, 243]) + sens

# Define a buffer to store the past 2 seconds of trajectory points
trajectory = deque(maxlen=10000)



# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.

result = cv2.VideoWriter(videoname + ' Ellipse Track.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         60, size)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the frame to get only the color pixels
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Apply Gaussian blur
    mask = cv2.GaussianBlur(mask, (5, 5), 1)  # (5, 5) kernel size, 0 standard deviation
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
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

        if len(max_contour) >= 5:  # FitEllipse requires at least 5 points
            # Find the ellipse that best fits the contour
            ellipse = cv2.fitEllipse(max_contour)
            (x_ellipse, y_ellipse), (major_axis, minor_axis), angle = ellipse

            # Draw the fitted ellipse
            cv2.ellipse(frame, ellipse, (0, 0, 255), 2)


            # Draw the center of the fitted ellipse
            cv2.circle(frame, (int(x_ellipse), int(y_ellipse)), 6, (255, 0, 0), -8)

            # Draw the trajectory of the fitted ellipse's center for the past 2 seconds
            trajectory.appendleft((int(x_ellipse), int(y_ellipse)))
            for i in range(1, len(trajectory)):
                if trajectory[i - 1] is None or trajectory[i] is None:
                    continue
                cv2.line(frame, trajectory[i - 1], trajectory[i], (200, 255, 255), 3)

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

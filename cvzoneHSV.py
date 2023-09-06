
import cv2
from cvzone.ColorModule import ColorFinder

cap = cv2.VideoCapture('raga safe.mp4')

myColorFinder = ColorFinder(True)
hsvVals = 'red'
# Read the first frame

ret, frame = cap.read()
if not ret:
    exit(1)

while True:
    # Find color

    imgColor, mask = myColorFinder.update(frame, hsvVals)

    # Display the frame and its corresponding mask
    imgColor = cv2.resize(imgColor, (0, 0), None, 0.7, 0.7)
    cv2.imshow("ImageColor", imgColor)
    key = cv2.waitKey(50)

    # Exit the loop if 'q' is pressed
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
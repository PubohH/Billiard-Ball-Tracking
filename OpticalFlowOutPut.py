import numpy as np
import cv2 as cv
import argparse
# parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
#                                               The example file can be downloaded from: \
#                                               https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# args = parser.parse_args()
videoname = 'jumpshot_no_collision' #input video clip name, no suffix needed
cap = cv.VideoCapture(videoname + '.mp4') #可以直接输入视频名称，或者输入args.image后用cmd运行
p0 = np.array([[[263, 302]]], np.float32) #需要追踪的像素点坐标，最好选择亮斑或和周围有明显差别的区域。
# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.

result = cv.VideoWriter(videoname + ' with trajectory.mp4',
                         cv.VideoWriter_fourcc(*'mp4v'),
                         60, size)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params) #寻找视频第一帧里值得追踪的特征角落，如果自行选点则无需使用
#cv.calcOpticalFlowPyrLK需要输入n维numpy array,所以选好点位后，需要如下输入坐标。

# p0 = np.array([p0[0]], np.float32)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params) #代码核心，光流法追踪。
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 4)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)
    result.write(img) #output with trajectory. don't put this after next line, because next line resize the window
    img = cv.resize(img, (0, 0), fx=0.6, fy=0.6)  # 将视频缩放，高=fy，宽=fx

    cv.imshow('frame', img)
    if cv.waitKey(1) == ord('q'):#每过1ms就会看键盘是否输入q，如果否，则进入下一次while循环，展示下一帧图片。
        break
    # k = cv.waitKey(30) & 0xff
    # if k == 27:
    #     break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
cv.destroyAllWindows()
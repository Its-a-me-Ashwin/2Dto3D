import numpy as np
import cv2
cap = cv2.VideoCapture(0)
ctr = 0
max_ctr = 100
while(True):
    # Capture frame-by-frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(gray, (8,6),None)
	if ret == True:
		print (ctr)
		cv2.imwrite("picture" + str(ctr) + ".jpg", frame)
		ctr = ctr + 1
		frame = cv2.drawChessboardCorners(frame, (8,6), corners, ret)
	if ctr > max_ctr:
		break
    # Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(100) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

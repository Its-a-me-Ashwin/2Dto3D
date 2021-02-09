from utils.arucolib import ArucoSingleTracker
from utils.o3dutils import *
from utils.transforms import *
import numpy as np
import cv2
import os

img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')

#print(img1.shape)
img1 = cv2.resize(img1,(640,480))
img2 = cv2.resize(img2,(640,480))

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)



id_to_find  = 6
marker_size  = 13.5 #- [cm]

#--- Get the camera calibration path
calib_path  = ""
camera_matrix   = np.loadtxt(calib_path+'cameraMatrix.txt', delimiter=',')
camera_distortion   = np.loadtxt(calib_path+'cameraDistortion.txt', delimiter=',')

aruco_tracker = ArucoSingleTracker(id_to_find=id_to_find, marker_size=marker_size, show_video=True, camera_matrix=camera_matrix, camera_distortion=camera_distortion)  


data1 = aruco_tracker.trackImage(img1)
data2 = aruco_tracker.trackImage(img2)



print("Image 1",data1)
print("Image 2",data2)


projection1,colors1 = projectImage(image=img1)
projection2,colors2 = projectImage(image=img2)


projection1 = rotate(data1[1][5],data1[1][4],data1[1][3],projection1)
#print(data1[1][3],data1[1][4],data1[1][5])
projection2 = rotate(data2[1][5],data2[1][4],data2[1][3],projection2)
print("Roatatiin done")

projection1 = translate(data1[1][0],data1[1][1],data1[1][2],projection1)
projection2 = translate(data2[1][0],data2[1][1],data2[1][2],projection2)
print("Trasnlatong done")

pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(projection1)
pcd1.colors = o3d.utility.Vector3dVector(colors1)

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(projection2)
pcd2.colors = o3d.utility.Vector3dVector(colors2)


d1 = list()
d1.append(float(data1[1][0]))
d1.append(float(data1[1][1]))
d1.append(float(data1[1][2]))


d2 = list()
d2.append(float(data2[1][0]))
d2.append(float(data2[1][1]))
d2.append(float(data2[1][2]))

#print(d1)
line1 = makeLine(d1,[0,0,0])
line2 = makeLine(d2,[0,0,0])

#print(line1)
draw(objects=[line1,line2],others = [pcd1,pcd2])


if False:
    files = os.listdir("./")
    for i,f in enumerate(files):
        if f.endswith("jpg"):
            img = cv2.imread(f)
            img = cv2.resize(img,(640,480))
            data = aruco_tracker.trackImage(img)
            if data:
                print(i,f)


if False:
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == False:
            break
        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        data = aruco_tracker.trackImage(frame)
        if data != None and data != False:
            print(data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

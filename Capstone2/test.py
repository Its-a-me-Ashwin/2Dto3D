from utils.arucolib import ArucoSingleTracker
from utils.o3dutils import *
from utils.transforms import *
import numpy as np
import cv2
import os
from math import pi,sqrt,acos


img1 = cv2.imread('./inputs/test6/_DSC0294.JPG')
img2 = cv2.imread('./inputs/test6/_DSC0295.JPG')

#print(img1,img2)

#print(img1.shape)
img1 = cv2.resize(img1,(4496,3000))
img2 = cv2.resize(img2,(4496,3000))

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

def absa(l):
    s = 0
    for i in l:
        s += i*i
    return sqrt(s)

print()

x = np.dot(data1[0][:3],data2[0][:3])/(absa(data2[0][:3])*absa(data1[0][:3]))
print(x)
print(acos(x)*(180/pi))
print(rotateMatrix(data1[0][3],data1[0][4],data1[0][5]))

projection1,colors1 = projectImage(image=img1)
projection2,colors2 = projectImage(image=img2)


projection1 = rotate(data1[1][3],data1[1][4],data1[1][5]+pi,projection1)
projection2 = rotate(data2[1][3],data2[1][4],data2[1][5]+pi,projection2)
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


if True:
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


from aruco3d import calMatrices,makeEsentialMatrix,cameraCoords2Real


p1 = data1[1]
p2 = data2[1]
cp1 = data1[2][0]
cp2 = data2[2][0]

R1,t1 = calMatrices(p1,p2)
R2,t2 = calMatrices(p2,p1)

# g1 = ThreeDPointToPixel(R1,t1,camDict,c1)

# g2 = ThreeDPointToPixel(R2,t2,camDict,c2)

# draw(objects=[p11,m1,m2,c1,c2,g1,g2],axis=False)

E = makeEsentialMatrix(R=R1, t=t1)

#print(cp1[0][1],cp2[0][1])

arP1 = np.array(cameraCoords2Real(camDict,[cp1]))

arP1 = rotate(p1[3],p1[3],p1[5],arP1)
arP1 = translate(p1[0],p1[1],p1[2],arP1)


arP2 = np.array(cameraCoords2Real(camDict,[cp2]))
arP2 = rotate(p2[3],p2[3],p2[5],arP2)
arP2 = translate(p2[0],p2[1],p2[2],arP2)

print(arP1.shape,arP2.shape)




diff = np.matmul(np.matmul(arP1,E),arP2.T)
print(E,arP1,arP2)
print(diff)

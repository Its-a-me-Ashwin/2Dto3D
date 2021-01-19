    
import numpy as np
import cv2
import cv2.aruco as aruco
import math
import functools 





def getCameraMatrix():
	with np.load('System.npz') as X:
		camera_matrix, dist_coeff, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
	return camera_matrix, dist_coeff


def sin(angle):
	return math.sin(math.radians(angle))

def cos(angle):
	return math.cos(math.radians(angle))

def distance (x,y):
        return int(math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)/2)

def angle (x,y):
        return abs((y[1]-x[1]/y[1]-x[1]))


def detect_markers(img, camera_matrix, dist_coeff):
        markerLength = 100
        aruco_list = []
	######################## INSERT CODE HERE ########################
        j=0
        aruco_dict=aruco.Dictionary_get(aruco.DICT_5X5_250)
        parameters=aruco.DetectorParameters_create()
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        corners,ids,_=aruco.detectMarkers(img_gray,aruco_dict,parameters=parameters)
        for i in corners:
                id_cur=ids[j]
                j+=1
                rvec, tvec, _= aruco.estimatePoseSingleMarkers(i,135,camera_matrix,dist_coeff)
                centerX=0
                centerY=0
                for x,y in i[0]:
                        centerX+=x
                        centerY+=y
                centerX/=4
                centerY/=4
                aruco_list.append((id_cur,(centerX,centerY),rvec,tvec))
                #print (aruco_list)
	##################################################################
        return aruco_list


def drawAxis(img, aruco_list, aruco_id, camera_matrix, dist_coeff):
	for x in aruco_list:
		if aruco_id == x[0]:
			rvec, tvec = x[2], x[3]
	markerLength = 100
	m = markerLength/2
	pts = np.float32([[-m,m,0],[m,m,0],[-m,-m,0],[-m,m,m]])
	pt_dict = {}
	imgpts, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeff)
	for i in range(len(pts)):
		 pt_dict[tuple(pts[i])] = tuple(imgpts[i].ravel())
	src = pt_dict[tuple(pts[0])];   dst1 = pt_dict[tuple(pts[1])];
	dst2 = pt_dict[tuple(pts[2])];  dst3 = pt_dict[tuple(pts[3])];
	
	img = cv2.line(img, src, dst1, (0,255,0), 4)
	img = cv2.line(img, src, dst2, (255,0,0), 4)
	img = cv2.line(img, src, dst3, (0,0,255), 4);print(list(map(lambda x: x*(180/3.1415),rvec[0][0])))
	return img


def drawCube(img, ar_list, ar_id, camera_matrix, dist_coeff):
        for x in ar_list:
                if ar_id == x[0]:
                        rvec, tvec = x[2], x[3]
        markerLength = 100
        m = markerLength/2
	######################## INSERT CODE HERE ########################
        pts = np.float32([[-m,m,0],[m,m,0],[-m,-m,0],[-m,m,m*2],[m,-m,m*2],[m,m,m*2],[-m,-m,m*2],[m,-m,0]])
        pt_dict = {}
        imgpts, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeff)
        imgpts = np.int32(imgpts).reshape(-1,2)
        for i in range(len(pts)):
                pt_dict[tuple(pts[i])] = tuple(imgpts[i].ravel())
        src = pt_dict[tuple(pts[0])];   dst1 = pt_dict[tuple(pts[1])];
        dst2 = pt_dict[tuple(pts[2])];  dst3 = pt_dict[tuple(pts[3])];
        dst4 = pt_dict[tuple(pts[4])];  dst5 = pt_dict[tuple(pts[5])];
        dst6 = pt_dict[tuple(pts[6])];  dst7 = pt_dict[tuple(pts[7])];
        img = cv2.line(img, src, dst1, (0,0,255), 4)
        img = cv2.line(img, src, dst2, (0,0,255), 4)
        img = cv2.line(img, dst7, dst4, (0,0,255), 4)
        img = cv2.line(img, src, dst3, (0,0,255), 4)
        img = cv2.line(img, dst5, dst3, (0,0,255), 4)
        img = cv2.line(img, dst5, dst1, (0,0,255), 4)
        img = cv2.line(img, dst5, dst4, (0,0,255), 4)
        img = cv2.line(img, dst2, dst7, (0,0,255), 4)
        img = cv2.line(img, dst2, dst6, (0,0,255), 4)
        img = cv2.line(img, dst6, dst4, (0,0,255), 4)
        img = cv2.line(img, dst6, dst3, (0,0,255), 4)
        img = cv2.line(img, dst7, dst1, (0,0,255), 4)
        #cv2.imwrite( "../SavedResults/drawCube/cube1.jpg", img );
        #################################################################
        return img


def drawCylinder(img, ar_list, ar_id, camera_matrix, dist_coeff):
        for x in ar_list:
                if ar_id == x[0]:
                        rvec, tvec = x[2], x[3]
        markerLength = 100
        radius = markerLength/2; height = markerLength*1.5
	######################## INSERT CODE HERE ########################
        m=radius
        h=height
        for x in ar_list:
                if ar_id == x[0]:
                        cords=x[1]
        pts = np.float32([[m,0,0],[m*cos(30),m*sin(30),0],[m*cos(60),m*sin(60),0],[0,m,0],[m*cos(120),m*sin(120),0],
                          [m*cos(150),m*sin(150),0],[-m,0,0],[m*cos(210),m*sin(210),0],[m*cos(240),m*sin(240),0],
                          [m*cos(270),m*sin(270),0],[m*cos(300),m*sin(300),0],[m*cos(330),m*sin(330),0],
                          [m,0,h],[m*cos(30),m*sin(30),h],[m*cos(60),m*sin(60),h],[0,m,h],[m*cos(120),m*sin(120),h],
                          [m*cos(150),m*sin(150),h],[-m,0,h],[m*cos(210),m*sin(210),h],[m*cos(240),m*sin(240),h],
                          [m*cos(270),m*sin(270),h],[m*cos(300),m*sin(300),h],[m*cos(330),m*sin(330),h],[0,0,h],[0,0,h]])




        pt_dict = {}
        imgpts, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeff)
        imgpts = np.int32(imgpts).reshape(-1,2)
        for i in range(len(pts)):
                pt_dict[tuple(pts[i])] = tuple(imgpts[i].ravel())

        
        dst0 = pt_dict[tuple(pts[0])];   dst1 = pt_dict[tuple(pts[1])];
        dst2 = pt_dict[tuple(pts[2])];  dst3 = pt_dict[tuple(pts[3])];
        dst4 = pt_dict[tuple(pts[4])];  dst5 = pt_dict[tuple(pts[5])];
        dst6 = pt_dict[tuple(pts[6])];  dst7 = pt_dict[tuple(pts[7])];
        dst8 = pt_dict[tuple(pts[8])];   dst9 = pt_dict[tuple(pts[9])];
        dst10 = pt_dict[tuple(pts[10])];  dst11 = pt_dict[tuple(pts[11])];



        dst12 = pt_dict[tuple(pts[12])];   dst13 = pt_dict[tuple(pts[13])];
        dst14 = pt_dict[tuple(pts[14])];  dst15 = pt_dict[tuple(pts[15])];
        dst16 = pt_dict[tuple(pts[16])];  dst17 = pt_dict[tuple(pts[17])];
        dst18 = pt_dict[tuple(pts[18])];  dst19 = pt_dict[tuple(pts[19])];
        dst20 = pt_dict[tuple(pts[20])];   dst21 = pt_dict[tuple(pts[21])];
        dst22 = pt_dict[tuple(pts[22])];  dst23 = pt_dict[tuple(pts[23])];

        c1 = pt_dict[tuple(pts[24])];  c2 = pt_dict[tuple(pts[25])];
        
        
        cv2.line(img, dst0, dst12, (255,0,0), 2)
        cv2.line(img, dst1, dst13, (255,0,0), 2)
        cv2.line(img, dst2, dst14, (255,0,0), 2)
        cv2.line(img, dst3, dst15, (255,0,0), 2)
        cv2.line(img, dst4, dst16, (255,0,0), 2)
        cv2.line(img, dst5, dst17, (255,0,0), 2)
        cv2.line(img, dst6, dst18, (255,0,0), 2)
        cv2.line(img, dst7, dst19, (255,0,0), 2)
        cv2.line(img, dst8, dst20, (255,0,0), 2)
        cv2.line(img, dst9, dst21, (255,0,0), 2)
        cv2.line(img, dst10, dst22, (255,0,0), 2)
        cv2.line(img, dst11, dst23, (255,0,0), 2)

        cv2.line(img, dst0, dst6, (255,0,0), 2)
        cv2.line(img, dst1, dst7, (255,0,0), 2)
        cv2.line(img, dst2, dst8, (255,0,0), 2)
        cv2.line(img, dst3, dst9, (255,0,0), 2)
        cv2.line(img, dst4, dst10, (255,0,0), 2)
        cv2.line(img, dst5, dst11, (255,0,0), 2)


        cv2.line(img, dst12, dst18, (255,0,0), 2)
        cv2.line(img, dst13, dst19, (255,0,0), 2)
        cv2.line(img, dst14, dst20, (255,0,0), 2)
        cv2.line(img, dst15, dst21, (255,0,0), 2)
        cv2.line(img, dst16, dst22, (255,0,0), 2)
        cv2.line(img, dst17, dst23, (255,0,0), 2)

        cv2.line(img, dst0, dst1, (255,0,0), 2)
        cv2.line(img, dst1, dst2, (255,0,0), 2)
        cv2.line(img, dst2, dst3, (255,0,0), 2)
        cv2.line(img, dst3, dst4, (255,0,0), 2)
        cv2.line(img, dst4, dst5, (255,0,0), 2)
        cv2.line(img, dst5, dst6, (255,0,0), 2)
        cv2.line(img, dst6, dst7, (255,0,0), 2)
        cv2.line(img, dst7, dst8, (255,0,0), 2)
        cv2.line(img, dst8, dst9, (255,0,0), 2)
        cv2.line(img, dst9, dst10, (255,0,0), 2)
        cv2.line(img, dst10, dst11, (255,0,0), 2)
        cv2.line(img, dst11, dst0, (255,0,0), 2)

        cv2.line(img, dst12, dst13, (255,0,0), 2)
        cv2.line(img, dst13, dst14, (255,0,0), 2)
        cv2.line(img, dst14, dst15, (255,0,0), 2)
        cv2.line(img, dst15, dst16, (255,0,0), 2)
        cv2.line(img, dst16, dst17, (255,0,0), 2)
        cv2.line(img, dst17, dst18, (255,0,0), 2)
        cv2.line(img, dst18, dst19, (255,0,0), 2)
        cv2.line(img, dst19, dst20, (255,0,0), 2)
        cv2.line(img, dst20, dst21, (255,0,0), 2)
        cv2.line(img, dst21, dst22, (255,0,0), 2)
        cv2.line(img, dst22, dst23, (255,0,0), 2)
        cv2.line(img, dst23, dst12, (255,0,0), 2)
        return img

if __name__=="__main__":
    cap = cv2.VideoCapture(0)
    cam, dist = getCameraMatrix()
    while(True):
        # Capture frame-by-frame
        ret, img = cap.read()

        aruco_list = detect_markers(img, cam, dist)
        for i in aruco_list:
                img = drawAxis(img, aruco_list, i[0], cam, dist)
                #img = drawCube(img, aruco_list, i[0], cam, dist)
                #img = drawCylinder(img, aruco_list, i[0], cam, dist)
        
        # Display the resulting frame
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    '''
    cam, dist = getCameraMatrix()
    img = cv2.imread("./TestCases/image_7.jpg")
    aruco_list = detect_markers(img, cam, dist)
    for i in aruco_list:
            img = drawAxis(img, aruco_list, i[0], cam, dist)
            img = drawCube(img, aruco_list, i[0], cam, dist)
            img = drawCylinder(img, aruco_list, i[0], cam, dist)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
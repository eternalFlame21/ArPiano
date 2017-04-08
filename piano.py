import cv2
import numpy as np
class ShapeDetector:
    def __init__(self):
        pass
    def thresholdEdge(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray=cv2.GaussianBlur(gray,(5,5),0)
        thresh = cv2.Canny(gray,100,200)
        kernel = np.ones((5,5),np.uint8)
        thresh = cv2.dilate(thresh,kernel,iterations = 1)
        return thresh

    def thresholdWhiteKeys(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray=cv2.GaussianBlur(gray,(5,5),0)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        kernel = np.ones((10,3),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return thresh
    def thresholdBlackKeys(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray=cv2.GaussianBlur(gray,(5,5),0)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        kernel = np.ones((5,3),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return ~thresh 

    def extractShapes(self, thresh,af=4.5,arLow=-0.1,arHigh=10,debug=False,debugWin='binary'):
        if(debug):
            cv2.imshow(debugWin,thresh)
        W,H=thresh.shape 
        (conts, _)=cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates=[]
        for cont in conts:
            approx=self.countVerticx(cont)
            (x, y, w, h) = cv2.boundingRect(approx)
            ar=w/float(h)
            if(len(approx)==4)and(cv2.contourArea(approx)>(W*H/af))and(ar>arLow)and(ar<arHigh):
                candidates.append(approx)
        candidates = sorted(candidates, key = cv2.contourArea, reverse = True)
        return candidates

    def countVerticx(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        return approx

def getx(cod):
    return cod[0]

def gety(cod):
    return cod[1]

def extractImg(img,cont):
    border=np.reshape(cont,(4,2))
    cods=[]
    border=sorted(border, key = getx)
    if border[0][1]<border[1][1]:
        cods.append(border[0])
        cods.append(border[1])
    else :
        cods.append(border[1])
        cods.append(border[0])
    if border[2][1]>border[3][1]:
        cods.append(border[2])
        cods.append(border[3])
    else:
        cods.append(border[3])
        cods.append(border[2])
    border=cods
    h,w,c=img.shape 
    M = cv2.getPerspectiveTransform(np.float32(border),np.float32([[0,0],[0,h],[w,h],[w,0]]))
    dst = cv2.warpPerspective(img, M, (w, h))
    img=cv2.flip(dst,0)
    return img


def capturePiano():
    sd=ShapeDetector()
    cam=cv2.VideoCapture(0)
    font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, .5, 0, 2, 1)
    while True:
        ret,img=cam.read()
        img=cv2.resize(img, (650,400), interpolation = cv2.INTER_AREA)
        piano=img.copy()
        h,w,c=piano.shape
       #getting the frame
        thresh=sd.thresholdEdge(img)
        candts=sd.extractShapes(thresh,debug=True,af=32,debugWin='frame',arLow=2.5)
        cv2.drawContours(img, candts, 0, (0,255,0), 3)
        if(len(candts)>0):
            piano=extractImg(piano,candts[0])
            border=np.reshape(candts[0],(4,2))
            for i in range(len(border)):
                cv2.cv.PutText(cv2.cv.fromarray(img),str(i)+":"+str(border[i]),(border[i][0],border[i][1]),font, (0,0,255))

            wPiano=piano[int(h*0.5):int(h*0.8),0:w]

            bPiano=piano[int(h*0.2):int(h*0.4),0:w]
            #getting white keys
            wth=sd.thresholdWhiteKeys(wPiano)
            wkCandts=sd.extractShapes(wth,af=20,arLow=0.2,arHigh=1,debug=True,debugWin='wk')
            cv2.drawContours(wPiano, wkCandts, -1, (255,0,0), 3)
            #getting black keys
            bth=sd.thresholdBlackKeys(bPiano)
            bkCandts=sd.extractShapes(bth,af=20,arLow=0.2,arHigh=1,debug=True,debugWin='bk')
            cv2.drawContours(bPiano, bkCandts, -1, (0,0,255), 3)
            cv2.imshow("piano",piano)
            #cv2.imshow('invPiano',invPiano)
        #if(len(wkCandts)==6)and(len(bkCandts)==5):
            #return candts[0],wkCandts,bkCandts
        cv2.imshow('result',img) 
        cv2.waitKey(10)



if __name__ == '__main__':
    capturePiano()

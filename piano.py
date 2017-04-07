import cv2
import numpy as np
class ShapeDetector:
    def __init__(self):
        pass
    def thresholdEdge(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.Canny(gray,100,200)
        return thresh

    def thresholdWhiteKeys(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        kernel = np.ones((10,3),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return thresh
    def thresholdBlackKeys(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        return approx

def main():
    sd=ShapeDetector()
    cam=cv2.VideoCapture(0)
    while True:
        ret,img=cam.read()
        img=cv2.resize(img, (650,400), interpolation = cv2.INTER_AREA)
        #getting the frame
        thresh=sd.thresholdEdge(img)
        candts=sd.extractShapes(thresh,debug=True,debugWin='frame',arLow=1.2)
        cv2.drawContours(img, candts, 0, (0,255,0), 3)
        if(len(candts)>0):
            (x,y,w,h)=cv2.boundingRect(candts[0])
            piano=img[y:y+h,x:x+w]
            wPiano=img[y+int(h/1.7):y+h,x:x+w]
            bPiano=img[y+int(h/5):y+int(h/2.5),x:x+w]
            #getting white keys
            wth=sd.thresholdWhiteKeys(wPiano)
            wkCandts=sd.extractShapes(wth,af=20,arLow=0.2,arHigh=0.5,debug=True,debugWin='wk')
            cv2.drawContours(wPiano, wkCandts, -1, (255,0,0), 3)
            #getting black keys
            bth=sd.thresholdBlackKeys(bPiano)
            bkCandts=sd.extractShapes(bth,af=50,arLow=0.2,arHigh=1,debug=True,debugWin='bk')
            cv2.drawContours(bPiano, bkCandts, -1, (0,0,255), 3)
            cv2.imshow("piano",piano)
            #cv2.imshow('invPiano',invPiano)

        cv2.imshow('result',img) 
        cv2.waitKey(10)

if __name__ == '__main__':
    main()

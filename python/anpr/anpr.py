from skimage.segmentation import clear_border
import pytesseract
import numpy as np
import imutils
import cv2
import ipdb


class ANPR:
    def __init__(self, minAR=4, maxAR=5, debug=False):
        self.minAR = minAR
        self.maxAR = maxAR
        self.debug = debug

    def debug_imshow(self, title, image, waitKey=True):
        if self.debug:
            cv2.imshow(title, image)

            if waitKey:
                cv2.waitKey(0)

    def locate_license_plate_candidates(self, gray, keep=5):
        print("inside locate_license_plate_candidates")

        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        self.debug_imshow("Blackhat", blackhat, True)
        cv2.imshow("Image", blackhat)
        cv2.waitKey(0)
        # ipdb.set_trace()
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(
            light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Light Regions after Otsi's IBT", light)
        cv2.imshow("Image", light)
        cv2.waitKey(0)

        # getting BB of the license plate using Scharr gradient rep
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255*((gradX-minVal)/(maxVal-minVal))
        gradX = gradX.astype("uint8")
        self.debug_imshow("Scharr", gradX)
        cv2.imshow("Image", gradX)
        cv2.waitKey(0)

        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(
            gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Grad Thresh", thresh)
        cv2.imshow("Image", thresh)
        cv2.waitKey(0)

        # erosions n dilations to denoise image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.debug_imshow("Grad Erode/Dilate", thresh)
        cv2.imshow("Image", thresh)
        cv2.waitKey(0)

        # bitwise and
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        self.debug_imshow("Final", thresh, waitKey=True)
        cv2.imshow("Image", thresh)
        cv2.waitKey(0)
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

        return cnts

    def locate_license_plate(self, gray, candidates, clearBorder=False):
        print("inside locate_license_plate")
        lpCnt = None
        roi = None
        for c in candidates:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w/float(h)
            #check is LP is rect
            if True:
                # if ar >= self.minAR and ar <= self.maxAR:
                lpCnt = c
                licensePlate = gray[y:y+h, x:x+w]
                roi = cv2.threshold(licensePlate, 0, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                if clearBorder:
                    roi = clear_border(roi)

                self.debug_imshow("License Plate", licensePlate)
                cv2.imshow("Image", licensePlate)
                cv2.waitKey(0)
                self.debug_imshow("ROI", roi, waitKey=True)
                cv2.imshow("Image", roi)
                cv2.waitKey(0)
                break
            print(roi, lpCnt)
        return (roi, lpCnt)

    def build_tesseract_options(self, psm=7):
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        options += " --psm {}".format(psm)
        return options

    def find_and_ocr(self, image, psm=7, clearBorder=False):
        lpText = None
        print("inside anpr")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Image", gray)
        cv2.waitKey(0)
        candidates = self.locate_license_plate_candidates(gray)
        print(candidates)
        (lp, lpCnt) = self.locate_license_plate(
            gray, candidates, clearBorder=clearBorder)

        if lp is not None:
            options = self.build_tesseract_options(psm=psm)
            lpText = pytesseract.image_to_string(lp, config=options)
            self.debug_imshow("License_Plate", lp)
            cv2.imshow("Image", lp)
            cv2.waitKey(0)
        return (lpText, lpCnt)

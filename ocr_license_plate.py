# from python.anpr import ANPR
from python.anpr.anpr import ANPR
from imutils import paths
import argparse
import imutils
import cv2
import ipdb


def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input directory of images")
ap.add_argument("-c", "--clear-border", type=int, default=-1,
                help="whether or to clear border pixels before OCR'ing")
ap.add_argument("-p", "--psm", type=int, default=7,
                help="default PSM mode for OCR'ing license plates")
ap.add_argument("-d", "--debug", type=int, default=-1,
                help="whether or not to show additional visualizations")
args = vars(ap.parse_args())

anpr = ANPR(debug=args["debug"] > 0)
imagePaths = sorted(list(paths.list_images(args["input"])))
print(imagePaths)
# ipdb.set_trace()
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # ipdb.set_trace()
    (lpText, lpCnt) = anpr.find_and_ocr(
        image, psm=args["psm"], clearBorder=args["clear_border"] > 0)
    print(lpCnt, lpText)

    if lpText is not None and lpCnt is not None:
        box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
        box = box.astype("int")
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

        (x, y, w, h) = cv2.boundingRect(lpCnt)
        cv2.put(image, cleanup_text(lpText), (x, y-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        print("{}".format(lpText))
        cv2.imshow("Output ANPR", image)
        cv2.waitKey(0)

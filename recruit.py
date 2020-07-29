from PIL import ImageGrab, Image
import pandas as pd
import pytesseract
import re
import cv2
import imutils

rx = re.compile(r'[A-Z][a-z]+')

csv = pd.read_csv(r'./RecruitmentTags.csv', header=None, names=[
	'Name',
	'Rank',
	'1st',
	'2nd', 
	'3rd',
	'type'
	])

class Recruit(object):
	TESSERACT_CONFIG = ("-l eng --oem 1 --psm 7")
	def __init__(self, img):
		self.img = cv2.imread(img)
		self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

	def image_to_string(self, img):
		return pytesseract.image_to_string(img, config=self.TESSERACT_CONFIG)

	def threshold(self):
		ret, thresh = cv2.threshold(self.gray, 100, 255, cv2.THRESH_BINARY)
		return thresh

	def detect_text(self, crop):
		im = imutils.resize(crop, height=200)
		im = cv2.bitwise_not(im)
		# im = cv2.bitwise_not(im)
		# im = cv2.bitwise_not(im)

		text = self.image_to_string(im)
		if len(text)>=3:
			return text

	def detect_rects(self):
		thresh = self.threshold()
		contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		for contour in contours:
			approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
			if len(approx)!=4:continue
			x, y, w, h = k = cv2.boundingRect(approx)
			if (w*h)<100:continue
			im = self.gray[y:y+h, x:x+w]
			yield im

	def get_tags(self):
		arr = set()
		for i in self.detect_rects():
			text = self.detect_text(i)
			if text:
				arr.add(text)
		return arr

if __name__ == '__main__':
	recruit = Recruit(r'Screenshot_20200501-103732.png')
	print(recruit.get_tags())

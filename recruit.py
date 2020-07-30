from PIL import ImageGrab, Image
import pandas as pd
import pytesseract
import re
import cv2
import imutils

rx = re.compile(r'[A-Z][a-z]+')

all_tags_col = [	
	'1st',
	'2nd', 
	'3rd',
	'4th',
	'5th']
csv = pd.read_csv(r'./RecruitmentTags.csv', header=None, names=[
	'name',
	'rank',
	'1st',
	'2nd', 
	'3rd',
	'4th',
	'5th',
	'type'
	])
csv = csv.where(pd.notnull(csv), None)
csv['tags'] = csv[all_tags_col].values.tolist()

class Recruit(object):
	TESSERACT_CONFIG = ("-l eng --oem 1 --psm 7")
	@staticmethod
	def all_recruitment_tags():
		return csv
	@classmethod
	def find_operators(cls, tags, mini=True):
		df = cls.all_recruitment_tags()
		def filter(x):
			i = tags.intersection(set(x))
			return list(i) if len(i) else None
		df['inter'] = df.tags.apply(filter)
		df['length'] = df['inter'].str.len()
		df = df.sort_values(['length', 'rank'], ascending=False)
		df = df[df.inter.notnull()]
		if mini:
			df = df[[i for i in df.columns if i not in all_tags_col]]

		return df

	def __init__(self, img):
		self.img = cv2.imread(img) if isinstance(img, str) else img
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

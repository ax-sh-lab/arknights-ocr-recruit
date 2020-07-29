import streamlit as st 

print = st.write

import recruit
import cv2
import imutils

r = recruit.Recruit(r'Screenshot_20200501-103732.png')
st.image(r.img, use_column_width=True, channels='BGR')
st.image(r.gray, caption='BW IMG',use_column_width=True)

# for i in r.detect_rects():
# 	st.image(i)
# 	text = r.detect_text(i)
# 	st.write(text)

st.write(r.get_tags())

# def detect_text_boxs(img):
# 	arr = []
# 	img = best_thresh(img)
# 	binary = cv2.bitwise_not(img)
# 	contours = r.find_contours(binary)
# 	# img = r.img
# 	st.image(binary, use_column_width=True)
# 	print(len(contours))
# 	for contour in contours:
# 		(x,y,w,h) = contour
# 		if w<h:continue
# 		# print(x, y)
# 		im = r.gray[y:y+h, x:x+w]
# 		im = recruit.imutils.resize(im, height=200)
# 		im = cv2.bitwise_not(im)
# 		im = cv2.bitwise_not(im)
# 		im = cv2.bitwise_not(im)

# 		text = recruit.pytesseract.image_to_string(im, config=config)
# 		if len(text)>=3:
# 			arr.append(text)
# 		# st.image(im)
# 	st.write(arr)

# # detect_text_boxs(r.gray)




# def find_rects(img):
# 	thresh = best_thresh(img)
# 	cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE,
# 		cv2.CHAIN_APPROX_NONE)

# 	for c in cnts:
# 		approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
# 		if len(approx)!=4:continue
# 		x, y, w, h = k = cv2.boundingRect(approx)
# 		if (w*h)<100:continue
# 		im = img[y:y+h, x:x+w]
# 		text = detect_text(im)
# 		st.write(text)
# 		st.image(im)
# 		# cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,0))
# 		# cv2.drawContours(img, [approx], 0, (0,0,255), 5)
# 		# r = approx.ravel()
# 		# x, y = r[:2]

# 	st.image(img)
# 	# st.image(thresh)






# find_rects(r.gray)

















# import numpy as np
# def box_detect(gray):
# 	dst = cv2.cornerHarris(gray,2,3,0.04) #result is dilated for marking the corners, not important
# 	dst = cv2.dilate(dst, None)
# 	ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
# 	dst = np.uint8(dst)
# 	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
# 	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# 	corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
# 	img = r.img
# 	for i in range(1, len(corners)):
# 		cv2.circle(img, (int(corners[i,0]), int(corners[i,1])), 7, (0,255,0), 2)
# 	st.image(img, use_column_width=True)

# # box_detect(r.gray)

# def text_det(img2gray):
#     thresh = best_thresh(img2gray)
#     thresh = cv2.dilate(thresh, (3,3), iterations=10)
#     text = image_to_string(thresh)
#     st.image(thresh)
#     st.write(text)
#     # ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
#     # image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
#     # ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
#     # '''
#     #         line  8 to 12  : Remove noisy portion 
#     # '''
#     # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
#     #                                                      3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
#     # dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation


#     # for cv2.x.x

#     # _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours

#     # # for cv3.x.x comment above line and uncomment line below

#     # #image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


#     # for contour in contours:
#     #     # get rectangle bounding contour
#     #     [x, y, w, h] = cv2.boundingRect(contour)

#     #     # Don't plot small false positives that aren't text
#     #     if w < 35 and h < 35:
#     #         continue

#     #     # draw rectangle around contour on original image
#     #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

#     #     '''
#     #     #you can crop image and send to OCR  , false detected will return no text :)
#     #     cropped = img_final[y :y +  h , x : x + w]

#     #     s = file_name + '/crop_' + str(index) + '.jpg' 
#     #     cv2.imwrite(s , cropped)
#     #     index = index + 1

#     #     '''
#     # # write original image with added contours to disk
#     # # cv2.imshow('captcha_result', img)
#     # # cv2.waitKey()


# # text_det(r.gray)








# #     cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)
# # thresh = recruit.imutils.skeletonize(r.gray, size=(3, 3))
# # thresh = cv2.Canny(r.gray, 50, 50)
# # # thresh = cv2.reode(thresh, None, iterations=1)
# # st.image(thresh, use_column_width=True)

# # # r.gray = recruit.imutils.resize(thresh, height=1080)




# # text = recruit.pytesseract.image_to_string(thresh, config=config)
# # print(text)

# # _, r.gray = cv2.threshold(r.gray,
# # 	200,
# # 	255,
# # 	cv2.THRESH_BINARY)

# # st.image(r.gray, use_column_width=True)

# # binary = cv2.bitwise_not(thresh)


# # st.image(binary, use_column_width=True)
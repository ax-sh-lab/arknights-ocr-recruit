import streamlit as st 

print = st.write
from recruit import Recruit
import numpy as np
import cv2

img = st.file_uploader('Recruit tag screenshot uploader', ['png', 'jpg', 'jpeg'])

if img is not None:
	st.image(img, use_column_width=True)
	img.seek(0)
	file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
	mat = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
	recruit = Recruit(mat)
	tags = recruit.get_tags()
	print(tags)

	# tags = {'Caster', 'Starter', 'Medic', 'Ranged', 'Guard'}
	df = Recruit.find_operators(tags)
	df[['rank',	'name', 'inter',  'length']]
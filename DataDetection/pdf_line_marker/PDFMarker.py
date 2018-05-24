# @author   huizhan

'''
    mark out the text lines in text box in the pdf pages, and convert into jpg images.
'''

import os

import cv2
from PIL import Image
from PDF2Image import dump
from PDF2Textline import extract


def marker(pdf_name):

	imgs = dump(pdf_name)
	pages = extract(pdf_name)

	bounded_imgs, pages_gt = [], []
	for img, page in zip(imgs, pages):

		# read img and page size
		img_size = img.shape[:2]
		img_height, img_width = img_size
		page_width = page['page_size']['width']
		page_height= page['page_size']['height']

		# cal the scale between img and page
		scale = img_width / page_width
		scale+= img_height/page_height
		scale/= 2

		# process the text line and generate gt file
		lines, boxes, page_gt = page['lines'], [], []
		for line in lines:
			box = [int(a*scale) for a in line[0]]
			box[1] = img_height - box[1]
			box[3] = img_height - box[3] 
			boxes.append(box)

			# generate line gt
			line_gt = [str(a) for a in box]
			line_gt.append(line[1])
			page_gt.append('\t'.join(line_gt))

		page_gt = '\n'.join(page_gt)
		pages_gt.append(page_gt)

		# draw bound box
		bounded_img = img.copy()
		for box in boxes:
			start, end = tuple(box[:2]), tuple(box[2:4])
			cv2.rectangle(bounded_img, start, \
				end, (0, 255, 0), thickness=2)

		bounded_img = Image.fromarray(bounded_img)
		bounded_imgs.append(bounded_img)

	imgs = [Image.fromarray(img) for img in imgs]

	return imgs, bounded_imgs, pages_gt



if __name__ == '__main__':


	pdf_root = './pdf/'
	pdfs = os.listdir(pdf_root)
	pdfs = [pdf for pdf in pdfs if \
		pdf.split('.')[-1] == 'pdf']


	target_root = './image/'
	if not os.path.exists(target_root):
		os.makedirs(target_root)
	file_id = os.listdir(target_root)
	file_id = [file for file in file_id \
		if file.split('.')[-1]=='jpg' and len(file.split('.'))==2]
	file_id = [int(file.split('.')[0]) for file in file_id]
	file_id = 1 if len(file_id) == 0 else max(file_id) + 1


	# process pdfs
	pdfs.sort()
	for pdf in pdfs[:]:
		print(pdf)

		pdf_path = pdf_root + pdf
		imgs, bounded_imgs, gts = marker(pdf_path)

		for img, bounded_img, gt in zip(imgs, bounded_imgs, gts):
			img.save(target_root + '%d.jpg' % file_id)
			bounded_img.save(target_root + '%d.bounded.jpg' % file_id)
			with open(target_root + '%d.jpg.line.gt' % file_id, 'w+') as f:
				f.write(gt)

			file_id+=1

		os.remove(pdf_path)

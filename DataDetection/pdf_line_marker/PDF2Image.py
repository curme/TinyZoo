# @author   huizhan

'''
    used to convert pdf file to images (numpy array).
'''

import io
import os

import cv2
import numpy
import PyPDF2
import skimage.io
from PIL import Image
from wand.color import Color
from wand.image import Image as WandImage


# convert pdf file to numpy array list
# code revised from stacko-verflow answer
#   https://stackoverflow.com/a/50141279/9238864
# thanks a lot
def dump(pdf_path):

    src_pdf = PyPDF2.PdfFileReader(open(pdf_path, "rb"))

    # What follows is a lookup table of page numbers within 
    # sample_log.pdf and the corresponding filenames.
    pages = src_pdf.getNumPages()

    imgs = []
    # convert each page to jpg
    for pageno in range(pages):

        # convert pdf page to wand image
        img = pdf_page_to_jpg(src_pdf, pagenum=pageno, resolution=300)
        img.format = 'jpg'

        # convert wand image to numpy array
        img_buffer=numpy.asarray(bytearray(img.make_blob()), dtype=numpy.uint8)
        bytesio = io.BytesIO(img_buffer)
        img = skimage.io.imread(bytesio)
        
        imgs.append(img)

    return imgs


# convert pdf page to wand image
# code revised from jrsmith3/pdfmod.py 
#   https://gist.github.com/jrsmith3/9947838
# thanks a lot
def pdf_page_to_jpg(src_pdf, pagenum = 0, resolution = 72):
    
    dst_pdf = PyPDF2.PdfFileWriter()
    dst_pdf.addPage(src_pdf.getPage(pagenum))

    pdf_bytes = io.BytesIO()
    dst_pdf.write(pdf_bytes)
    pdf_bytes.seek(0)

    img = WandImage(file = pdf_bytes, resolution = resolution)
    img.convert("jpg")

    return img


if __name__ == '__main__':

    pdf_path = './pdf/限定增强数据集.pdf'
    imgs = dump(pdf_path)

    for i in range(len(imgs)):
        img = Image.fromarray(imgs[i])
        img.save('%d.jpg' % i)
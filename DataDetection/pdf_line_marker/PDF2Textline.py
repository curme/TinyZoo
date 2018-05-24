# @author	huizhan

'''
	used to extract textboxes in the pdf.
'''

from pdfminer.layout import LAParams
from pdfminer.layout import LTTextBox
from pdfminer.layout import LTTextLine
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFParser
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfdocument import PDFDocument


# extract text boxes in the pdf file
# code revised from official document:
#	https://media.readthedocs.org/pdf/pdfminer-docs/latest/pdfminer-docs.pdf
# thanks a lot
def extract(pdf_path, password=''):

	# open pdf file
	fp = open(pdf_path, 'rb')
	parser = PDFParser(fp)
	document = PDFDocument(parser, password)
	if not document.is_extractable: raise PDFTextExtractionNotAllowed

	# Create a PDF page aggregator object.
	laparams = LAParams()
	rsrcmgr = PDFResourceManager()
	device = PDFPageAggregator(rsrcmgr, laparams=laparams)
	interpreter = PDFPageInterpreter(rsrcmgr, device)

	# extract 
	pages = []
	for page in PDFPage.create_pages(document):

		# page size
		_, _, width, height = page.mediabox
		page_size = {'width':width, 'height':height}

		# interpret the page
		interpreter.process_page(page)
		layout = device.get_result()

		# extract text lines in the page
		text_boxes = [obj for obj in layout \
			if isinstance(obj, LTTextBox)]

		page_textlines = []
		for text_box in text_boxes:
			page_textlines += [info(obj) for obj in \
			text_box if isinstance(obj, LTTextLine)]

		page = {'page_size':page_size}
		page['lines'] = page_textlines
		pages.append(page)


	return pages


def info(line):

	# position
	x0, y0 = line.x0, line.y0
	x1, y1 = line.x1, line.y1
	position = [x0, y0, x1, y1]

	# text
	text = line.get_text()
	if text[-2:] == ' \n':
		text = text[:-2]

	return (position, text)


if __name__ == '__main__':

	pdf_path = './pdf/限定增强数据集.pdf'
	extract(pdf_path)

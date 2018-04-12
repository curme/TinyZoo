# CRNN

A simple demo for paper [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717).  
The main purpose of this program is demonstrate the crnn algorithm in a clear and light way. Containing only the training process.

## Program Structure
The project mainly consists of three file:   
 `preprocess.py`, `model.py` and `main.py`,   
responsible to prepare data batch, build model and control process respectively.

## How to run
Prepare Fonts:  
Download the [font files](https://www.dropbox.com/s/p3bdia4xe0dok7i/Fonts.zip?dl=0) from dropbox. (~ 233.3M)  
Extract the font files zip to `PROJECTPATH/Static/Fonts/`.

Configure Chars:  
Configure the characters you want to examine in the configure char map block of `preprocess.py` (~ line_no:50) . 

Run Demo:  
Run `python main.py` in your command line.
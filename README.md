---
# Quick OCR
## Motivation
To make an easy to implement OCR from scratch
## Requirements
Python 3.8 or above with all [requirements](requirements.txt) dependencies installed. To install run:
```python
$ pip3 install -r requirements.txt
```
## To run scratch version
```python
$ python3 scratch.py
```
## To run tesseract version (works better)
```python
$ python3 tess.py
```
## Details:
#### scratch:

uses the mnist dataset and a_z handwritten alphabets dataset from kaggle
uses vgg16
accuracy after 10 epochs ~ 98%

#### tess:

uses tesseract instead of training from scratch
works way better


## Might Do
- [ ] Try Different OpenCV filters
----


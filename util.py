from PIL import Image, ImageTk
from cairosvg import svg2png
import _pickle as cPickle
import numpy as np
import os
from tkinter import filedialog


def parse_fn(fn):
    rev_fn = fn[::-1]
    rev_fn_slash_i = rev_fn.index('/')
    rev_fn_particle = rev_fn[2:rev_fn_slash_i]
    true_fn = rev_fn_particle[::-1]
    print(true_fn)
    return true_fn

def open_file():
    filename = filedialog.askopenfilename(initialdir='./pkl/training', title= 'Select training variation')


###########################
def set_img(img_fn, widget, img_fp='./img/', img_fx='.png'):
    ''' Load a PNG file, prepare it, and update the widget to display it. '''
    img = ImageTk.PhotoImage(Image.open(img_fp+img_fn+img_fx))
    widget.configure(image=img)
    widget.image = img

def prep_img(img_fn, img_fp='./img/', img_fx='.png'):
    return ImageTk.PhotoImage(Image.open(img_fp+img_fn+img_fx))

###########################


def from_pkl(fn, fp='./pkl/', fx='.p'):
    ''' Returns the contents of a pickled file as a Python object. '''
    with open(fp+fn+fx, 'rb') as f:
        result = cPickle.load(f)
        f.close()
    return result

##############################################################################

def to_pkl(obj, fn, fp='./pkl/', fx='.p'):
    ''' Stores the contents of a Python object in a pickled file. '''
    with open(fp+fn+fx, 'wb') as f:
        cPickle.dump(obj, f)
        f.close()

##############################################################################

def extant(fname, fp='./pkl', fx='.p'):
    f = fp + fname + fx
    if os.path.isfile(f):
        return True
    else:
        return False

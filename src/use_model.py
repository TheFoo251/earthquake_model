#Imports
import os
import PIL.Image
import PIL.ImageTk
import keras
import tensorflow as tf
import glob
import random
import cv2
from random import shuffle
import sys
from tkinter import *
from tkinter.filedialog import askopenfilename
import numpy as np


class MatPlotLibGUI:

    def __init__(self):

        print("  initializing the Helper class ...")
        self.root = Tk()
        self.root.title("Model")

        # initialize grid
        self.grid = Frame(self.root)
        self.grid.grid_columnconfigure(4, weight=1)
        self.grid.grid_rowconfigure(2, weight=1)

        # add buttons and labels
        self.load_image_button = Button(self.grid, text="Load Image", command=self.load_image)
        self.load_image_button.grid(row=1, column=1)

        self.image_name = Label(self.grid, text="none")
        self.image_name.grid(row=1, column=2, sticky='e')

        self.load_model_button = Button(self.grid, text="Load Model", command=self.load_model)
        self.load_model_button.grid(row=1, column=3)

        self.model_name = Label(self.grid, text="none")
        self.model_name.grid(row=1, column=4, sticky='e')

        # Create frames for images
        self.image_label = Label(self.grid)
        self.image_label.grid(row=2, column=1, columnspan=2)

        self.prediction_label = Label(self.grid)
        self.prediction_label.grid(row=2, column=3, columnspan=2)

        self.grid.pack()
        print("  done init!")

    def load_model(self, model_path):
        model = keras.models.load_model(model_path, custom_objects=None, compile=True)

    def load_image(self):
        path = askopenfilename()
        self.image_name.config(text = os.path.basename(path))
        raw = PIL.Image.open(path)
        og_img = PIL.ImageTk.PhotoImage(raw.resize((256, 256)))
        self.image_label.configure(image=og_img)
        self.image_label.image = og_img
        raw = np.array(raw.resize((256, 256)))/255.
        raw = raw[:,:,0:3]


    def read_from_file(self):
        """
        Spawn a dialogue to choose a file and read data from it
        """

    def run(self):
        print("    Entering the Tk main event loop")
        self.root.mainloop()
        print("    Leaving the Tk main event loop")





"""
TESTING (using hold data)
@TODO -- actually use hold data instead of testing image...
"""



"""


#predict the mask
pred = model.predict(np.expand_dims(raw, 0))

#mask post-processing
msk  = pred.squeeze()
msk = np.stack((msk,)*3, axis=-1)
msk[msk >= 0.5] = 1
msk[msk < 0.5] = 0

#show the mask and the segmented image
combined = np.concatenate([raw, msk, raw* msk], axis = 1)
plt.axis('off')
plt.imshow(combined)
plt.show()
"""

if __name__ == '__main__':

    print("Inside main...")
    gui = MatPlotLibGUI()
    gui.run()
    print("done!")
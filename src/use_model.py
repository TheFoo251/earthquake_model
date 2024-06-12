# Imports
import os
import PIL.Image
import PIL.ImageTk
import sys
from tkinter import *
from tkinter.filedialog import askopenfilename
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tensorflow as tf


class MatPlotLibGUI:
    def __init__(self):
        print("  initializing the Helper class ...")
        self.root = Tk()
        self.root.title("Model")

        # initialize grid
        self.grid = Frame(self.root)
        self.grid.grid_columnconfigure(4, weight=1)
        self.grid.grid_rowconfigure(5, weight=1)

        # Row 1
        self.load_model_button = Button(
            self.grid, text="Load Model", command=self.load_model
        )
        self.load_model_button.grid(row=1, column=1)
        self.model_name = Label(self.grid, text="none")
        self.model_name.grid(row=1, column=2, sticky="e")

        # Row 2
        self.load_image_button = Button(
            self.grid, text="Load Image", command=self.load_image
        )
        self.load_image_button.grid(row=2, column=1)
        self.image_name = Label(self.grid, text="none")
        self.image_name.grid(row=2, column=2, sticky="e")

        self.predict_button = Button(
            self.grid, text="Run Prediction", command=self.run_prediction
        )
        self.predict_button.grid(row=2, column=3)

        # Row 3
        # Create frames images
        self.image_label = Label(self.grid)
        self.image_label.grid(row=3, column=1, columnspan=2)

        # Row 4
        self.prediction_frame = Frame(self.grid)
        self.prediction_frame.grid(row=4, column=1, columnspan=4)

        # Row 5
        self.quit_button = Button(self.grid, text="QUIT", command=self.quit)
        self.quit_button.grid(row=5, column=1)

        self.grid.pack()

        # Create empy img and mask
        self.raw_img = []
        self.mask = []
        self.model = []

        print("  done init!")

    def quit(self):
        sys, exit()

    def load_image(self):
        path = askopenfilename()
        self.image_name.config(text=os.path.basename(path))
        raw = PIL.Image.open(path)
        og_img = PIL.ImageTk.PhotoImage(raw.resize((256, 256)))
        self.image_label.configure(image=og_img)
        self.image_label.image = og_img
        raw = np.array(raw.resize((256, 256))) / 255.0
        raw = raw[:, :, 0:3]
        self.raw_img = raw

    def load_model(self):
        path = askopenfilename()
        self.root.config(cursor="watch")
        self.model_name.config(text="loading model...")
        self.model = tf.keras.models.load_model(path, custom_objects=None, compile=True)
        self.root.config(cursor="")
        self.model_name.config(text=os.path.basename(path))
        self.model.summary()
        # print(self.model.get_weights())

    def run_prediction(self):
        # predict the mask
        print("predicting")  # replace with loading bar
        pred = self.model.predict(np.expand_dims(self.raw_img, 0))
        print("prediction:", pred)  # DEBUG

        # mask post-processing
        msk = pred.squeeze()
        msk = np.stack((msk,) * 3, axis=-1)
        msk[msk >= 0.5] = 1
        msk[msk < 0.5] = 0

        damaged_pixels = msk.sum()
        # @TODO -- replace w/labels..
        print("mask:", msk)
        print("damaged pixels:", damaged_pixels)

        combined = np.concatenate([msk, self.raw_img * msk], axis=1)

        # @DEBUG -- try to show image with matplotlib...

        f = plt.Figure()
        a = f.add_subplot(111)
        plt.axis("off")
        a.imshow(combined)

        # Re-create plot_frame so we only have one plot showing at a time

        self.prediction_frame = Frame(self.grid)
        self.prediction_frame.grid(row=4, column=1, columnspan=4)

        canvas = FigureCanvasTkAgg(f, master=self.prediction_frame)
        canvas.draw()  # drawing the new plot in frame canvas

        canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
        canvas._tkcanvas.pack(side="top", fill="both", expand=1)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = MatPlotLibGUI()
    gui.run()

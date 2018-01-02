from main import main_func
from main import loadImage
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2

def select_image():
    global panelA, panelB

    path = filedialog.askopenfilename()

    if len(path) > 0:
        # main_func(path)
        image1 = loadImage(path)
        image2 = main_func(path)
        # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        image1 = Image.fromarray(image1)
        image2 = Image.fromarray(image2)
        image1 = ImageTk.PhotoImage(image1)
        image2 = ImageTk.PhotoImage(image2)
        if panelA is None or panelB is None:
            panelA = Label(image=image1)
            panelA.image = image1
            panelA.pack(side="left", padx=10, pady=10)
            panelB = Label(image=image2)
            panelB.image = image2
            panelB.pack(side="right", padx=10, pady=10)
        else:
            panelA.configure(image=image1)
            panelB.configure(image=image2)
            panelA.image = image1
            panelB.image = image2


root = Tk()
panelA = None
panelB = None

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# kick off the GUI
root.mainloop()

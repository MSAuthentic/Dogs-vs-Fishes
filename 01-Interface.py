import tensorflow_text as text
from tkinter import font
from tkinter.constants import ANCHOR, CENTER, X
import tensorflow as tf
from tensorflow import keras
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import cv2
from time import sleep
import numpy as np

IMAGE_HEIGHT, IMAGE_WIDTH= 190,190

def select_file():
    global x
    filetypes = (
        ('Image', '*.png *.jpeg *.jpg'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open File',
        initialdir='/',
        filetypes=filetypes)

    img = cv2.imread(filename=filename)
    cv2.imshow('Selected Image', img)
    sleep(2)
    x=filename
    if(filename ==''):
        showinfo('Image','No image selected!')

        
def CaricaModello():
    try:
        localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

        new_model = tf.keras.models.load_model(r'C:\Users\ankit\Desktop\GitHub\Cani-vs-Pesci\Dogs-vs-Fishes\Models\Modello_Cani_Pesci',options=localhost_save_option )       
        return new_model
    except Exception as e:
        print(e)
        showinfo('Model Error','Error While Loading the Model')
        
        
def  TestaImmagine():
    try:
   
        img = tf.keras.utils.load_img(
        x, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
            )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        model= CaricaModello()
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])      
        CLASS_NAMES= ['Cani','Pesci']
        showinfo('Risultato' ,
                "This image looks like in the category of {} with  {:.2f} of confidence percentage."
                .format(CLASS_NAMES[np.argmax(score)], 100 * np.max(score)))
    except Exception as e:
        print(e)  
        
        
        
        
def CreaGUI(root):
    root.title('Dog or Fish?')
    root.geometry("500x200")
    root.resizable(False,False)
    frm = tk.Frame(root, background='white', height=200, width=500)
    frm.place(relx=0,rely=0)

    lb= ttk.Label(frm,
                text="Select an Image!",
                justify='center',
                font=("Footlight MT Light",20),
                background='white')
    lb.place(relx=0.5,
            rely=0.2,
            anchor=CENTER)

    btnLoadImg = tk.Button(frm, text='Select Image',
                            command=select_file,
                            background='#212529',
                            foreground='white',
                            width=16,
                            height=2,
                            justify='center')
    btnLoadImg.place(relx=0.2, rely=0.5,anchor=CENTER)


    btnPredictImg =  tk.Button(frm, text='Guess the Image',
                            command= TestaImmagine,
                            background='#212529',
                            foreground='white',
                            width=16,
                            height=2,
                            justify='center')
    btnPredictImg.place(relx=0.48, rely=0.5,anchor=CENTER)


    btnQuit =  tk.Button(frm, text="Exit",
                        command=root.destroy,
                        background='#212529',
                        foreground='white',
                        width=16,
                        height=2)

    btnQuit.place(relx=0.78,
                rely=0.5,
                anchor=CENTER)
    lbLogs= ttk.Label(frm, text="LOGS",justify='center',font=("Footlight MT Light",15), background='white')
    lbLogs.place(relx=0.1, rely=0.9, anchor=CENTER)  
    
    
           
root  = tk.Tk()
CreaGUI(root=root)

root.mainloop()
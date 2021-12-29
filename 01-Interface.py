import imp
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
        ('Immagine', '*.png *.jpeg *.jpg'),
        ('Tutti i files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Apri un file',
        initialdir='/',
        filetypes=filetypes)

    img = cv2.imread(filename=filename)
    cv2.imshow('Immagine Selezionata', img)
    sleep(2)
    x=filename
    if(filename ==''):
        showinfo('Immagine','Nessun immagine selezionata!')

        
def CaricaModello():
    try:
        new_model = tf.keras.models.load_model(r'C:\Users\User\Desktop\Models\Modello_Cani_Pesci')       
        return new_model
    except Exception as e:
        showinfo('Modello','Errore durante il caricamento')
        
        
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
                "Questa immagine sembra essere parte della categoria {} con  {:.2f} percento di confidenza."
                .format(CLASS_NAMES[np.argmax(score)], 100 * np.max(score)))
    except Exception as e:
        print(e)  
        
        
        
        
def CreaGUI(root):
    root.title('Cane o Pesce?')
    root.geometry("500x200")
    root.resizable(False,False)
    frm = tk.Frame(root, background='white', height=200, width=500)
    frm.place(relx=0,rely=0)

    lb= ttk.Label(frm,
                text="Selezionare un'immagine!",
                justify='center',
                font=("Footlight MT Light",20),
                background='white')
    lb.place(relx=0.5,
            rely=0.2,
            anchor=CENTER)

    btnLoadImg = tk.Button(frm, text='Scegli Immagine',
                            command=select_file,
                            background='#212529',
                            foreground='white',
                            width=16,
                            height=2,
                            justify='center')
    btnLoadImg.place(relx=0.2, rely=0.5,anchor=CENTER)


    btnPredictImg =  tk.Button(frm, text='Indovina l''immagine',
                            command= TestaImmagine,
                            background='#212529',
                            foreground='white',
                            width=16,
                            height=2,
                            justify='center')
    btnPredictImg.place(relx=0.48, rely=0.5,anchor=CENTER)


    btnQuit =  tk.Button(frm, text="Esci",
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
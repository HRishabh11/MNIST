#GUI
import tkinter as Tk
from tkinter import *
from tkinter import filedialog, messagebox, Button, LabelFrame, Canvas
from PIL import ImageGrab
import cv2
import numpy as np
import os
import keras

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
class Model:
    def __init__(self):
        pass
    
    def predict(self):
        img = cv2.imread('image.png')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(28,28),interpolation = cv2.INTER_AREA)
        cv2.imwrite("im2.png", img)
        img = img.astype('float32')
        img /= 255
        img = np.reshape(img,(1,28,28,1))
        model = keras.models.load_model('model.h5')
        pred_values = model.predict(img)
        pred_values = np.argmax(pred_values,axis = 1)    
        return pred_values[0]
        
class MainApp:
    def __init__(self):
        self.pre = None
        #model to predict
        self.model = Model()
        #creating root
        self.root = Tk()
        self.root.title("Digit Recognizer")
        self.root.resizable(0,0)
        #Gui elements
        self.lbl_drawhere = LabelFrame(text = 'Draw Here With Mouse')
        self.area_draw = Canvas(self.lbl_drawhere, width = 504,
                                height = 504, bg ='black')
        self.area_draw.bind('<B1-Motion>',self.draw)
        
        self.btn_reset = Button(self.lbl_drawhere, text = "Reset Drawing",
                                bg = "lightblue",command = self.reset_drawing)
        self.btn_predict = Button(self.root, text = 'Predict Digit',
                                  bg = "blue", command = self.predict_digit)
        
        #Fitting in th Gui
        self.lbl_drawhere.pack(in_=self.root, side = LEFT, fill = X)
        self.area_draw.pack()
        self.btn_reset.pack()
        self.btn_predict.pack(in_=self.root, side = LEFT)
        
    def draw(self,event):
        self.area_draw.create_oval(event.x,event.y,event.x+27,event.y+27,
                                   outline='white',fill = 'white')
        self.area_draw.create_rectangle(event.x,event.y,event.x+25,event.y+25,
                                        outline = 'white',fill = 'white')
        self.pre = 'D'
    
    def run(self):
        self.root.mainloop()
        
    def reset_drawing(self):
        self.area_draw.delete('all')
    
    def predict_digit(self):
        if(self.pre == None):
            messagebox.showerror(title = 'No Images',
                                 messgae = 'First Draw a number')
        else:
            x = self.root.winfo_rootx() + self.area_draw.winfo_x()
            y = self.root.winfo_rooty() + self.area_draw.winfo_y()
            x1 = x + self.area_draw.winfo_width()
            y1 = y + self.area_draw.winfo_height()
            ImageGrab.grab(bbox = (x,y+10,x1,y1)).save('image.png')
            messagebox.showinfo(
                    title = 'Prediction' , 
                    message = "Number: {}".format(self.model.predict()))
            
if __name__ == '__main__':
    MainApp().run()
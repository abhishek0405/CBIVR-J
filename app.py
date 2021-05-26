from flask import Flask, render_template, request,send_from_directory
import requests
import numpy as np
from model import autoencoder_clothes
import os
from cv2 import cv2
import pathlib
import glob

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app=Flask(__name__)


@app.route('/',methods=['GET', 'POST'])
def Home():
    if request.method == 'POST':
        file1 = request.files['file1']
        print(file1)

        
        
        if file1:
            filename1 = file1.filename
            target = os.path.join(APP_ROOT, 'images\\')
            print(target)
        
            if not os.path.isdir(target):
                os.mkdir(target)
            directory_current = pathlib.Path.cwd()
            img_fol = (os.path.join(directory_current, "images/"))
            if len(os.listdir(img_fol) ) != 0:
                os.remove(os.path.join(directory_current, "images/test.jpg"))
            destination = "\\".join([target, filename1])
            file1.save(destination)
            d= os.path.join(target, "test.jpg")
            os.rename(destination, d)
            image= cv2.imread( d, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (300, 300),interpolation = cv2.INTER_AREA)
            image=np.array(image, dtype='float')
            image /= 255
            image_names = os.listdir('test_results')
            print(len(image_names))
            num_images =len(image_names)//2
            print(num_images)
            
            autoencoder_clothes.retrieve_closest_images(image, 0, len(image_names)//2)



            
            

        return render_template('index1.html', image_name1='original_image'+str(num_images)+'.jpg', image_name2='retrieved_results'+str(num_images)+'.jpg')

        
            
    else:
        return render_template('index1.html',image_name1='', image_name2='')
    
@app.route('/<filename>')
def send_image(filename):
    return send_from_directory("test_results", filename)



if __name__=="__main__":
    app.run(debug=True)
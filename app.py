#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

# Ensure the application uses only the CPU
tf.config.set_visible_devices([], 'GPU')

from flask import Flask, request, render_template
import uuid
import os
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
from io import BytesIO

ALLOWED_EXTENSION = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNELS = 3

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSION

app = Flask(__name__)
model = load_model('CNN_100.h5', compile=True)

@app.route('/')
def index():
    return render_template('ImageML.html')

@app.route('/api/image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('ImageML.html', prediction='No posted image. Should be attribute named image')
    
    file = request.files['image']
    if file.filename == '':
        return render_template('ImageML.html', prediction='You did not select an image')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print("***" + filename)
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = Image.open(BytesIO(file.read()))
        img.load()
        img = img.resize((100, 100))
        img = np.reshape(img, [1, 100, 100, 3])
        image = tf.cast(img, tf.float32)
        cl = model.predict(image)
        indices = cl.argmax()
        
        if indices == 1:
            return render_template('ImageML.html', prediction='bleached_corals')
        else:
            return render_template('ImageML.html', prediction='healthy_corals')
    else:
        return render_template('ImageML.html', prediction='Invalid File extension')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

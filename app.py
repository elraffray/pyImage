from imageclassifier import ImageClassifier
from flask import Flask, render_template, request
import logging
import time
import cv2
import random
import glob
import os 

logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)


old_target = None

@app.route('/', methods=['GET', 'POST'])
def root():
    global old_target
    for fl in glob.glob('static/res_*.png'):
        os.remove(fl)

    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        nb_clusters = request.form['clusters']
        if request.form['effect'] == 'sort' and nb_clusters == '':
            return render_template('index.html')
        img = request.files['img']
        if img.filename == '':
            if old_target == None:
                return render_template('index.html')
            in_name = old_target
        else:    
            for fl in glob.glob('static/target_*'):
                os.remove(fl)
            ext = img.mimetype.split('/')[1]
            in_name = "target_ " + str(time.time()) + "." + ext
            img.save('static/' + in_name)
            old_target = in_name
        
        dst_name = "res_" + str(time.time()) + ".png"

        if request.form['effect'] == 'sort':
            classifier = ImageClassifier(int(nb_clusters), 'static/' + in_name)
            classifier.run('static/' + dst_name)
        elif request.form['effect'] == 'rand_row':
            randomize('static/' + in_name, 'static/' + dst_name)
        elif request.form['effect'] == 'rand_col':
            randomize('static/' + in_name, 'static/' + dst_name, cols=True)
        
        return render_template('index.html', path_in=in_name, path_out=dst_name)

def randomize(path_in, path_out, cols=False):
    img = cv2.imread(path_in)

    if cols:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    rows,_,_ = img.shape
    out = img.copy()
    new_idx = [i for i in range(rows)]
    random.shuffle(new_idx)

    for i in range(rows):
        out[i] = img[new_idx[i]]
    
    if cols:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imwrite(path_out, out)
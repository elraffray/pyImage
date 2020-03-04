from imageclassifier import ImageClassifier
from flask import Flask, render_template, request
import logging
import time

logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def root():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        nb_clusters = request.form['clusters']
        img = request.files['img']
        ext = img.mimetype.split('/')[1]
        path = "static/target." + ext
        img.save(path)
        dst_name = "res_" + str(time.time()) + ".png"


        if request.form['effect'] == 'sort':
            classifier = ImageClassifier(int(nb_clusters), path)
            classifier.run('static/' + dst_name)
        



        return render_template('index.html', img=dst_name)
        
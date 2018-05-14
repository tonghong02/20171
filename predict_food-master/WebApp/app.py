import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from predict import predict

from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__)) # duong dan tuong doi

@app.route("/")
def index():
    filename = 'static/pho.jpg'
    data,name = predict(filename)
    return render_template("index.html", image_name='pho.jpg', datas = data,names=name)

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/') # quy dinh thu muc chua anh : images

    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))

    # for upload in request.files.getlist("file"):
    
    upload=request.files.getlist("file")[0] # lay file anh
    filename = upload.filename # lay ten file anh
    destination = "/".join([target, filename]) # quy dinh duong dan ve thuc muc chua anh
    upload.save(destination)      
    data,name = predict('images/'+filename)
    return render_template("index.html",image_name=filename,datas=data,names=name)
    

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename=filename)
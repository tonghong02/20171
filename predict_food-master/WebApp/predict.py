from keras.models import load_model
from keras.preprocessing import image

import numpy as np
import json


# result = {}
result=[]
name=''
# dimensions of our images
img_width, img_height = 96, 96
#
# load the model we saved
model = load_model('../models/models.h5')

# predicting images
def predict(path):
    result = []
    img = image.load_img(path, target_size=(img_width, img_height)) #load anh
    x = image.img_to_array(img) # chuyen anh ve dang ma tran
    x = np.expand_dims(x, axis=0)
    x /=255.0  # chuan hoa anh ve dang o-1

    classes = model.predict_classes(images, batch_size=10) # dua ra luon anh ten la gi , dua ra id cua anh
    pred = model.predict(images) # dua ra % ty le anh day thuoc vao lop anh nao
    print (pred)

    with open('../data.txt') as json_file:
        data = json.load(json_file)
        i=0
        for p in data['data']:
            result.append({
                    'id': p['id'],
                    'name': p['name'],
                    'acc' : pred[0][i]*100
                })
            i +=1
        for p in data['data']:
            if(p['id'] == classes):
                name = p['name']
   
    return (result,name)


# if __name__ == '__main__':
#     predict('images/cream.jpg')
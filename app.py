import numpy as np
import os
import uuid
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub

from flask import Flask, request
from flask_restful import Resource, Api





img_height = 224
img_width = 224

model = tf.keras.models.load_model('./Food_model_new.h5', custom_objects={'KerasLayer': hub.KerasLayer})
class_names = ['Bamemoodang', 'Gangkeawwan', 'Guayjab', 'Hoitod', 'Joke', 'Kaipadmedmamoung', 'Kaogangkaree', 
'Kaokaijeaw', 'Kaokamoo', 'Kaokanamoogrob', 'Kaokugkapi', 'Kaomogkai', 'Kaomoocrob', 'Kaomootod', 'Kaomunkai', 'Kaonanue', 'Kaonaped',
'Kaopadkrateam', 'Kaopadpoo', 'Kaopudkung', 'Kaopudpongkaree', 'Kaosoi', 'Kaoyumkaisab', 'Kapaomoo', 'Kuakai', 'Kungobwuncen', 'Moomanao', 
'Moopudking', 'Padseiew', 'Padthai', 'Phalo', 'Radnamoo', 'Salad', 'Somtum', 'Suki', 'Taohusongkrueng', 'Tomyumkung', 'Yentafour', 'Yummama', 'Yummooyor']

calories = [
405 ,63 ,540, 452, 89, 180, 247, 175, 35, 78, 475, 60, 200, 311, 500, 112, 397, 90, 
90, 90, 186, 70, 240, 140, 63, 810, 508, 200, 280, 120, 285, 55, 120, 127, 52, 455,
690, 410, 729, 550]

#Reference1=['https://www.calforlife.com','https://www.lovefitt.com','https://www.nutritionix.com/','https://www.calforlife.com','https://www.calforlife.com/',
#'https://www.calforlife.com/','https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com',
#'https://www.lovefitt.com/','https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com',
#'http://iandmybody.com/','http://iandmybody.com/','http://iandmybody.com/','http://iandmybody.com/','http://iandmybody.com/',
#'http://event.sanook.com/health/calories/','http://event.sanook.com/health/calories/','http://event.sanook.com/health/calories/','http://event.sanook.com/health/calories/','http://event.sanook.com/health/calories/',
#'https://www.honestdocs.co/table-of-calories-in-food-types','https://www.calforlife.com','https://www.calforlife.com','http://event.sanook.com/health/calories/','https://www.calforlife.com',
#'https://www.fatnever.com/calories/','https://www.honestdocs.co/table-of-calories-in-food-types','https://www.calforlife.com/','https://www.calforlife.com/','https://www.calforlife.com/',
#'https://www.fatnever.com/calories/','https://www.fatnever.com/calories/','https://www.fatnever.com/calories/','https://www.honestdocs.co/table-of-calories-in-food-types','https://www.fatnever.com/calories/',
#'https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com','https://www.fatnever.com/calories/','https://www.honestdocs.co/table-of-calories-in-food-types',
#'https://www.wongnai.com','https://www.fatnever.com/calories//','https://www.wongnai.com','https://www.wongnai.com','https://www.wongnai.com']

#Reference2=['-','-','-','-','-',
#'-','-','-','-','-',
#'-','-','-','-','-',
#'-','-','-','-','-',
#'-','-','-','-','-',
#'-','-','-','-','-',
#'-','-','-','-','-',
#'-','-','-','-','-',
#'-','-','-','-','-',
#'-','-','-','-','-',]

app = Flask(__name__)
api = Api(app)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, D):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        extension = os.path.splitext(file.filename)[1]
        f_name = str(uuid.uuid4()) + extension
        file.save(os.path.join('/app/upload/', f_name))
        
        img = keras.preprocessing.image.load_img(
            '/app/upload/' + f_name, target_size=(img_height, img_width)
        )

        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        print(predictions)
        score = predictions[0]

        index = np.argmax(score)
        class_name = class_names[index]
        cal = calories[index]
        #ref1 = Reference1[index]
        #ref2 = Reference2[index]
        score = 100*np.max(score)
        #score = score[index]
        

        #return json.dumps({'a': predictions[0].tolist()})
        return json.dumps({'class':class_name, 'score': score ,'calories':cal}, cls=DecimalEncoder )
        

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
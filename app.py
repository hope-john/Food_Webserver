import numpy as np
import os
import uuid
import json
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from flask import Flask, request
from flask_restful import Resource, Api

img_height = 224
img_width = 224

model = tf.keras.models.load_model('./pizza_model')
class_names = [ '3_Layer_Fried_Pork', 'Apple' ,'Bacon', 'Baked_Spinach_Cheese' ,'Banana',
 'Bao', 'Bolona', 'Cabbagefriedshrimp', 'Chayen', 'Crispywonton',
 'Curryfriedrice', 'Dragonfruit' ,'Eclair' ,'Frenchfries', 'Fried_Chicken',
 'Friedegg' ,'Friedporkribs', 'Ganghjeudmara', 'Grillbanana', 'Grilled_Pork',
 'Grilledstickyrice', 'Grillsquid', 'Guayjab', 'Guichai', 'Gyouza', 'Hoitod',
 'Hormok', 'Joke' ,'Ka_Pao_Moo', 'Kaijork', 'Kang_Jud_Taohuu_Moo_Sub',
 'Kang_Keaw_Wan' ,'Kanomjib' ,'Kanomkeng' ,'Kanomsaizai', 'Kanomtom',
 'Kao_Fried_Egg', 'Kao_Ka_Moo' ,'Kao_Kug_Kapi' ,'Kao_Mogg_Kai' ,'Kao_Moo_Crob',
 'Kao_Mun_Kai' ,'Kao_Ped_Yang' ,'Kaopadpoo' ,'Kapong_Tod_Nampra',
 'Kashade_Fried_Vegetable' ,'Khanombueang' ,'Khanomkhrok', 'Khaotommud',
 'Kluaikhaek' ,'Kungobwuncen' ,'Larb_Moo' ,'Lookchinping' ,'Maggaroni',
 'Mangostickyrice', 'Moo_Ma_Now' ',Moo_Sub_Dok_Mai_Jeen' ,'Moopudking',
 'Moowan' ,'Mooyor' ,'Muntomkhing' ,'Orange', 'Padthai' ,'Pananggai' 'Pancake',
 'Papaya' ,'Patonggo', 'Phalo' ,'Plapao', 'Popcorn' ,'Popeayuan',
 'Pumpkincustard' ,'Ricefriedgarlic', 'Roastedegg' ,'Roti', 'Saiou',
 'Sakutuadum', 'Salad', 'Sausage', 'Shimp_Fried_Rice', 'Somtum',
 'Spaghetti_Carbonara', 'Spicyfish' ,'Stirfriedcrispyporkkale', 'Strawberry',
 'Tabwan' ,'Taohusongkrueng', 'Tempurashrimp', 'Thaisoyfriednoddles', 'Toast',
 'Todmun' ,'Tom_Yum_Kung', 'Tomjabchai' ,'Tongmuan', 'Tubtimgrob', 'Waffle',
 'Watermelon' ,'Wun', 'Yentafour','Yum_Mama','no_data']

calories = [
     199,0 ,290
 ]

app = Flask(__name__)
api = Api(app)

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
        score = tf.nn.softmax(predictions[0])

        index = np.argmax(score)
        class_name = class_names[index]
        cal = calories[index]
        score = 100 * np.max(score)
        
        return json.dumps({'class':class_name, 'score': score ,'calories':calories})

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
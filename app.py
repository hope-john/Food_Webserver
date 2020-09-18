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

model = tf.keras.models.load_model('./food_model')
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
405 ,63 ,540, 452, 89, 180, 247, 175, 35, 78, 475, 60, 200, 311, 500, 112, 397, 90, 
90, 90, 186, 70, 240, 140, 63, 810, 508, 200, 280, 120, 285, 55, 120, 127, 52, 455,
690, 410, 729, 550, 596, 551, 559, 580, 393, 185, 60, 100, 285, 128, 591, 234, 30,
533, 325, 150, 225, 80, 645, 341, 240, 60, 450, 181, 227, 43, 290, 535, 150, 375, 
409, 300, 645, 77, 297, 420, 176, 140, 290, 595, 90, 742, 297, 516, 60, 210, 300,
80, 585, 313, 230, 61, 874, 140, 264, 291, 30, 67, 352, 215]

Reference1=['https://www.calforlife.com','https://www.lovefitt.com','https://www.nutritionix.com/','https://www.calforlife.com','https://www.calforlife.com/',
'https://www.calforlife.com/','https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com',
'https://www.lovefitt.com/','https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com',
'http://iandmybody.com/','http://iandmybody.com/','http://iandmybody.com/','http://iandmybody.com/','http://iandmybody.com/',
'http://event.sanook.com/health/calories/','http://event.sanook.com/health/calories/','http://event.sanook.com/health/calories/','http://event.sanook.com/health/calories/','http://event.sanook.com/health/calories/',
'https://www.honestdocs.co/table-of-calories-in-food-types','https://www.calforlife.com','https://www.calforlife.com','http://event.sanook.com/health/calories/','https://www.calforlife.com',
'https://www.fatnever.com/calories/','https://www.honestdocs.co/table-of-calories-in-food-types','https://www.calforlife.com/','https://www.calforlife.com/','https://www.calforlife.com/',
'https://www.fatnever.com/calories/','https://www.fatnever.com/calories/','https://www.fatnever.com/calories/','https://www.honestdocs.co/table-of-calories-in-food-types','https://www.fatnever.com/calories/',
'https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com','https://www.fatnever.com/calories/','https://www.honestdocs.co/table-of-calories-in-food-types',
'https://www.wongnai.com','https://www.fatnever.com/calories//','https://www.wongnai.com','https://www.wongnai.com','https://www.wongnai.com',
'https://www.fatnever.com/calories/','https://www.calforlife.com/','https://www.calforlife.com/','https://www.calforlife.com/','https://www.calforlife.com/',
'https://www.fatnever.com/calories/','https://www.fatnever.com/calories/','https://www.fatnever.com/calories/','https://www.fatnever.com/calories/','https://www.fatnever.com/calories/',
'https://www.wongnai.com','https://www.wongnai.com','https://www.wongnai.com','https://www.wongnai.com','https://www.wongnai.com',
'https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com',
'https://www.fatnever.com/calories/','https://www.fatnever.com/calories/','https://www.fatnever.com/calories/','https://www.fatnever.com/calories/','https://www.fatnever.com/calories/',
'https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com',
'https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com','https://www.calforlife.com',
'https://www.lovefitt.com/','https://www.lovefitt.com/','https://www.lovefitt.com/','https://www.lovefitt.com/','https://www.lovefitt.com/',
'https://www.lovefitt.com','https://www.lovefitt.com','https://www.lovefitt.com','https://www.lovefitt.com','https://www.lovefitt.com',
'https://www.calforlife.com/','https://www.calforlife.com/','https://www.calforlife.com/','https://www.calforlife.com/','https://www.priceza.com']

Reference2=['-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-',
'-','-','-','-','-']

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

        index = np.argmax(score)
        class_name = class_names[index]
        cal = calories[index]
        ref1 = Reference1[index]
        ref2 = Reference2[index]
        score = 100 * np.max(score)
        
        return json.dumps({'class':class_name, 'score': score ,'calories':cal ,'Reference1': ref1, 'Reference2':ref2 })

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
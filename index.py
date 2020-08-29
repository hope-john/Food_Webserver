# import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

img_height = 180
img_width = 180

model = tf.keras.models.load_model('./pizza_model')

print('\n\n\n')

# Check its architecture
# model.summary()

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
 'Watermelon' ,'Wun', 'Yentafour', 'Yum_Mama' ]

target_dir = './pizza/'
images = os.listdir(target_dir)

for img_path in images:
    target_image_path = target_dir + img_path
    # print(img)
    img = keras.preprocessing.image.load_img(
        target_image_path, target_size=(img_height, img_width)
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "{} is belongs to {} with a {:.2f} percent confidence."
        .format(img_path, class_names[np.argmax(score)], 100 * np.max(score))
    )
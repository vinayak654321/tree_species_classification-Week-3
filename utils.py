import tensorflow as tf
import numpy as np
from PIL import Image

# ✅ Tree Species Class Names (in order of training dataset)
class_names = [
    'Acer Capillipes', 'Acer Circinatum', 'Acer Japonicum', 'Acer Palmatum', 'Acer Rubrum',
    'Acer Rufinerve', 'Acer Saccharinum', 'Alnus Cordata', 'Alnus Maximowiczii',
    'Alnus Rubra', 'Alnus Sieboldiana', 'Alnus Viridis', 'Arundinaria Simonii',
    'Betula Austrosinensis', 'Betula Pendula', 'Callicarpa Bodinieri', 'Castanea Sativa',
    'Catalpa Bignonioides', 'Celtis Koraiensis', 'Cercis Siliquastrum', 'Cornus Chinensis',
    'Cornus Controversa', 'Cornus Macrophylla', 'Cotinus Coggygria', 'Crataegus Monogyna',
    'Cyclobalanopsis Glauca', 'Eucalyptus Globulus', 'Fagus Sylvatica', 'Ginkgo Biloba',
    'Ilex Aquifolium', 'Juglans Regia'
]

#  Prediction Function
def predict_species(image):
    model = tf.keras.models.load_model('model.h5')  # Make sure model.h5 is in the same folder
    image = image.resize((180, 180))  # Resize to match model input
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) / 255.0  # Normalize

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    label = class_names[predicted_class]

    return label, confidence

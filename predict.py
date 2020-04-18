import sys
import argparse
from PIL import Image
# print ('Number of arguments:', len(sys.argv), 'arguments.')
# print ('Argument List:', str(sys.argv))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('image_path', type=str, nargs='+', help='image path')
parser.add_argument('saved_model', type=str, help='model_path')
parser.add_argument('--top_k',  type=int, help='model_path')

args = parser.parse_args()
image_path = args.image_path
saved_model = args.saved_model

if args.top_k == None:
    top_k = 5
else:
    top_k = args.top_k
    
# TODO: Make all necessary imports.
import tensorflow as tf
import os
import tensorflow_datasets
import numpy
import time
import tensorflow_hub as hub
from matplotlib import pyplot
import json
from tensorflow.keras import layers
tensorflow_datasets.disable_progress_bar()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if len(image_path) == 1:
    image_path = image_path[0]
else:
    print('Input error')
    image_path = None

# TODO: Load the Keras model
# reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath)
reloaded_keras_model = tf.keras.models.load_model(saved_model)
print('successfully loaded model ', saved_model)
# print(list(reloaded_keras_model.signatures.keys()))
# reloaded_keras_model.summary()

# TODO: Plot 1 image from the training set. Set the title 
# of the plot to the corresponding class name. 
with open('label_map.json', 'r') as file_handle:
    class_names = json.load(file_handle)
    
# TODO: Create the process_image function

def process_image (array3d_image_original):
    array3d_image_processed = tf.image.resize(array3d_image_original, (224,224))
    array3d_image_processed /= 255
    array3d_image_processed = array3d_image_processed.numpy()
    return array3d_image_processed


# TODO: Create the predict function

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = numpy.asarray(im)
    processed_test_image = process_image(test_image)
    
    n_select = top_k
    ps = model.predict(processed_test_image[numpy.newaxis,:,:,:])
    array_probability = ps[0]
    array_index = numpy.flipud(numpy.argsort(array_probability))
    classes = []
    probs = []
    for i_index in array_index[:n_select]:
        classes.append(str(i_index))
        probs.append(array_probability[i_index])
    return probs, classes


def predict_and_print(image_path, top_k):
    im = Image.open(image_path)
    test_image = numpy.asarray(im)
    # pyplot.figure(figsize = (8,3), dpi = 200, facecolor = 'white')
    processed_test_image = process_image(test_image)
    # pyplot.subplot(121)
    # pyplot.imshow(processed_test_image)
    # pyplot.subplot(122)
    probs, classes = predict(image_path, reloaded_keras_model, top_k)
    # pyplot.barh(range(len(probs)), width = probs) # , width = 0.8)
    # pyplot.title('class probability')
    # print(probs, classes)
    top_class_names = []
    for str_class in classes:
        top_class_names.append(class_names[str(int(str_class) + 1)])
    for i_pred in range(len(probs)):
        print('Probability: {:.3f}, Name: {}'.format(probs[i_pred], top_class_names[i_pred]))
    # pyplot.yticks(range(len(classes)), top_class_names)
    # pyplot.ylim([-0.5, 4.5])
    # pyplot.xlim([0,1])
    # pyplot.tight_layout()
    # pyplot.show()
    return None


predict_and_print(image_path, top_k)



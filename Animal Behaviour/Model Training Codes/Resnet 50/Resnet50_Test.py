# NOTE:-
# 1. Make sure that all the libraries that are needed to run the program/train the model are
# installed/imported properly in your system. If so NO install it using pip install command
# 2. In this code i have Given the directory according to my local machine, so feel free to
# modify the directory according to where your dataset is present
# 3. And the last predict() functon calls here is for various test cases make sure to comment
# everything initially and run the cases one by one

import cv2
import numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.models import model_from_json

def predict_(image_path):
    # Load the Model from Json File
    json_file = open(r"C:/Users/ksvib/Downloads/Mini Project - 7th Sem/Practical Implementations/ResNet 50/model.json",
                     'r')
    model_json_c = json_file.read()
    json_file.close()
    model_c = model_from_json(model_json_c)

    # Load the weights
    model_c.load_weights(
        r"C:/Users/ksvib/Downloads/Mini Project - 7th Sem/Practical Implementations/ResNet 50/best_model.h5")

    # Compile the model
    opt = SGD(lr=1e-4, momentum=0.9)
    model_c.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Load the image you want to classify
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    image = cv2.resize(image, (224, 224))

    # Display the image using cv2.imshow
    cv2.imshow("Input Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Predict the image
    preds = model_c.predict(np.expand_dims(image, axis=0))[0]

    if preds[0] < 0.5:
        print("Predicted Label: Cat")
    else:
        print("Predicted Label: Dog")

predict_(r"C:/Users/ksvib/Downloads/Mini Project - 7th Sem/Dataset/dataset_new/testing_set/cats/cat.9922.jpg")

predict_(r"C:\Users\ksvib\Downloads\Mini Project - 7th Sem\Dataset\dataset_new\testing_set\dogs\dog.9980.jpg")

predict_(r"C:\Users\ksvib\Downloads\Mini Project - 7th Sem\Dataset\dataset_new\testing_set\dogs\dog.9953.jpg")

predict_(r"C:\Users\ksvib\Downloads\Mini Project - 7th Sem\Dataset\dataset_new\testing_set\cats\cat.9759.jpg")
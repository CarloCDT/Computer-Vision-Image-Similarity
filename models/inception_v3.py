import numpy as np
import tensorflow as tf

class Inception_V3:
    """
    Creates an inception v3 model
    """
    
    # Initializatoin
    def __init__(self, weights="imagenet", classifier_activation=None):
        self.model = tf.keras.applications.InceptionV3(weights=weights, classifier_activation=classifier_activation)

    # Predict Classes
    def predict_classes(self, image_path, top=10, print_results = True):

        """
        Print predicted clases for a given image using inceptionV3

        Parameters
        ----------
        image_path : str
            The file location

        """

        # Load and process the input image
        image = tf.keras.preprocessing.image.load_img(image_path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        reshape_image = tf.image.resize(image_arr, size=(299,299), method='bilinear')
        input_inception = tf.keras.applications.inception_v3.preprocess_input(np.array([reshape_image]))

        # Make model predictions
        predictions = self.model.predict(input_inception)
        prediction_set = set()

        # Print predictions
        if print_results:
            print("Top predictions: \n")
        for idx, pred in enumerate(tf.keras.applications.inception_v3.decode_predictions(predictions, top=top)[0]):
            prediction_set.add(pred[1].replace("_"," ").title())
            if print_results:
                print("{:02d}. {} - Score: {:.2f}".format(idx + 1,
                                                          pred[1].replace("_"," ").title(), 
                                                          pred[2]))
        return prediction_set
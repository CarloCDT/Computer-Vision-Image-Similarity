import numpy as np
import pandas as pd
import tensorflow as tf
import os
from IPython.display import Image as Image_display

class Inception_V3:
    """
    Creates an inception v3 model
    """
    
    # Inicialization
    def __init__(self, weights="imagenet", classifier_activation=None):
        self.model = tf.keras.applications.InceptionV3(weights=weights, classifier_activation=classifier_activation)
        
    def predict(self, input_inception):
        return self.model.predict(input_inception)
        
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
    
    
    
    def find_similar_images_dl(self, image1_name, top_k=3):
    
        # Initialize DataFrame
        df = pd.DataFrame(columns=["from", "to", "dissimilarity"])

        # Process Image
        input_inception = Inception_V3.preprocess_image(image_path = image1_name)
        root_tensor = self.model.predict(input_inception)[0]

        # Get all Images
        all_images = [f for f in os.listdir("images/") if f.split(".")[1]=="jpg"]

        for image2_name in all_images:
                if image1_name != image2_name:
                    input_image2 = Inception_V3.preprocess_image(image_path = image2_name)
                    output_tensor = self.model.predict(input_image2)[0]
                    distance = float(tf.norm(root_tensor-output_tensor, ord='euclidean'))

                    # Create Table
                    dtf_data = pd.DataFrame(data=[[image1_name, image2_name, distance]], columns=["from", "to", "dissimilarity"])
                    df = pd.concat([df, dtf_data])

        similar_images_df = df.sort_values(by="dissimilarity").reset_index(drop=True).head(top_k)

        url = os.path.join("images", image1_name)

        print("New Image:")
        display(Image_display(url))

        print("Similar Images")

        for row in similar_images_df.iterrows():

            print("Dissimilarity Score: {:.2f}".format(row[1]["dissimilarity"]))

            url = os.path.join("images", row[1]["to"])
            display(Image_display(url))
        
        
    
    @staticmethod
    def preprocess_image(image_path):
        #Process Image
        image = tf.keras.preprocessing.image.load_img(os.path.join("images",image_path))
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        reshape_image = tf.image.resize(image_arr, size=(299,299), method='bilinear')
        input_inception = tf.keras.applications.inception_v3.preprocess_input(np.array([reshape_image]))
        return input_inception
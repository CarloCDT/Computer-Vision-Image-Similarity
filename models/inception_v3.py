import numpy as np
import pandas as pd
import tensorflow as tf
import os
from IPython.display import Image as Image_display
from boto.s3.connection import S3Connection
import boto3
from PIL import Image
from tqdm import tqdm
from scipy.spatial.distance import cosine
import pickle

class Inception_V3:
    """
    Creates an inception v3 model
    """
    
    # Inicialization
    def __init__(self, ACCESS_ID, ACCESS_KEY, weights="imagenet", classifier_activation=None):
        self.model = tf.keras.applications.InceptionV3(weights=weights, classifier_activation=classifier_activation)
        self.ACCESS_ID = ACCESS_ID
        self.ACCESS_KEY = ACCESS_KEY
   
    def predict(self, input_inception):
        return self.model.predict(input_inception)
        
    # Predict Classes
    def predict_classes(self, file_name, top=15, print_results = True):

        """
        Print predicted clases for a given image using inceptionV3

        Parameters
        ----------
        image_path : str
            The file location

        """

        # Load and process the input image
        #image = tf.keras.preprocessing.image.load_img(image_path)
        input_inception = self.preprocess_image(image_path = file_name)

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
    
    def find_similar_images_dl(self, image1_name, metric='euclidean',top_k=3):
    
        # Initialize DataFrame
        df = pd.DataFrame(columns=["from", "to", "dissimilarity"])

        # Process Image
        #input_inception = self.preprocess_image(image_path = image1_name)
        img = tf.image.decode_image(open(os.path.join("images",image1_name), 'rb').read(), channels=3)
        reshape_image = tf.image.resize(img, size=(299,299), method='bilinear')
        input_inception = tf.keras.applications.inception_v3.preprocess_input(np.array([reshape_image]))
        
        root_tensor = self.model.predict(input_inception)[0]

        # Get all Images
        all_images = self.get_file_names(self.ACCESS_ID, self.ACCESS_KEY)
        
        # Read JSON
        image_tensors_name = 'image_tensors.pkl'
        if image_tensors_name in os.listdir("labels"):
            with open(os.path.join("labels", image_tensors_name), 'rb') as fp:
                image_tensor = pickle.load(fp)
        else:
            image_tensor = {}
        
        # Loop all the images
        for image2_name in tqdm(all_images):
                if image1_name != image2_name:
                    
                    if image2_name in image_tensor:
                        output_tensor = image_tensor[image2_name]
                    
                    else:
                        input_image2 = self.preprocess_image(image_path = image2_name)
                        output_tensor = self.model.predict(input_image2)[0]
                        image_tensor[image2_name] = output_tensor
                                              
                    if metric == "cosine":
                        distance = cosine(root_tensor, output_tensor)
                    else:
                        distance = float(tf.norm(root_tensor-output_tensor, ord=metric))

                    # Create Table
                    dtf_data = pd.DataFrame(data=[[image1_name, image2_name, distance]], columns=["from", "to", "dissimilarity"])
                    df = pd.concat([df, dtf_data])
                    
        #print(df)
        
        # Save Image Tensor
        with open(os.path.join("labels", image_tensors_name), 'wb') as fp:
            pickle.dump(image_tensor, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        similar_images_df = df.sort_values(by="dissimilarity").reset_index(drop=True).head(top_k)

        url = os.path.join("images", image1_name)

        print("New Image:")
        display(Image_display(url))

        print("Similar Images")

        for row in similar_images_df.iterrows():

            print("Dissimilarity Score: {:.2f}".format(row[1]["dissimilarity"]))

            #url = os.path.join("images", row[1]["to"])
            #display(Image_display(url))
            display(self.read_image(row[1]["to"], self.ACCESS_ID, self.ACCESS_KEY))
       
    @staticmethod
    def read_image(image_name, ACCESS_ID, ACCESS_KEY, bucket='carlo-computer-vision-project'):
        s3 = boto3.resource('s3', aws_access_key_id=ACCESS_ID, aws_secret_access_key= ACCESS_KEY)
        bucket = s3.Bucket(bucket)
        object = bucket.Object(image_name)
        response = object.get()
        file_stream = response['Body']
        img = Image.open(file_stream)
        return img
    
    @staticmethod
    def get_file_names(ACCESS_ID, ACCESS_KEY, bucket='carlo-computer-vision-project'):
        conn = S3Connection(ACCESS_ID, ACCESS_KEY)
        bucket = conn.get_bucket(bucket, validate=False)
        all_files = [k.name for k in bucket.list()]
        return all_files
        
    def preprocess_image(self, image_path):
        #Process Image
        image = self.read_image(image_name=image_path, ACCESS_ID=self.ACCESS_ID, ACCESS_KEY=self.ACCESS_KEY, bucket='carlo-computer-vision-project')
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        reshape_image = tf.image.resize(image_arr, size=(299,299), method='bilinear')
        input_inception = tf.keras.applications.inception_v3.preprocess_input(np.array([reshape_image]))
        return input_inception
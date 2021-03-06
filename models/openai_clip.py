import torch
import clip
import json
import cv2
from numba import cuda
from PIL import Image
from boto.s3.connection import S3Connection
import boto3
import numpy as np

class OpenAI_clip:
    """
    Creates an OpenAI Clip model
    """
    
    # Initializatoin
    def __init__(self, ACCESS_ID, ACCESS_KEY):
        # Load the model
        # pip install git+https://github.com/openai/CLIP.git
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load('ViT-B/32', self.device)
        except:
            #cuda.get_current_device().reset()
            print("Error!")
            self.model, self.preprocess = clip.load('ViT-B/32', self.device)
            
        # Load Imafenet 1000 labels
        with open('imagenet_1000_labels.json', 'r') as fp:
            imagenet_1000_labels = json.load(fp)
            self.imagenet_labels = list(imagenet_1000_labels.values())
            self.imagenet_labels = [a.split(",")[0] for a in self.imagenet_labels]
            
        self.ACCESS_ID = ACCESS_ID
        self.ACCESS_KEY = ACCESS_KEY

    # Predict Classes
    def predict_classes(self, image_path, labels=None, top=15, print_results = True):
        """
        Print predicted clases for a given image using OpenAI Clip

        Parameters
        ----------
        image_path : str
            The file location

        labels: list[str]
            list containing the possible labels

        top: int
            Number of labels to select

        print_results: boolean
            Print the top results

        """
        if not labels:
            labels = self.imagenet_labels
        
        # Transform image to PIL
        #img = cv2.imread(image_path)
        img = self.read_image(image_name=image_path, ACCESS_ID=self.ACCESS_ID, ACCESS_KEY=self.ACCESS_KEY, bucket='carlo-computer-vision-project')
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        im_pil

        # Prepare the inputs
        image_input = self.preprocess(im_pil).unsqueeze(0).to(self.device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in labels]).to(self.device)

        # Calculate features
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

        # Pick the top most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)#.softmax(dim=-1)
        values, indices = similarity[0].topk(min(top,similarity[0].shape[0]))

        prediction_set = set()

        # Print predictions
        if print_results:
            print("Top predictions: \n")
        for idx, (value, index) in enumerate(zip(values, indices)):
            prediction_set.add(labels[index].title())
            if print_results:
                print("{:02d}. {} - Score: {:.2f}".format(idx + 1,
                                              labels[index].title(), 
                                              value.item()))
        return prediction_set
    
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
    def get_file_names(ACCESS_ID, ACCESS_KEY, bucket):
        conn = S3Connection(ACCESS_ID, ACCESS_KEY)
        bucket = conn.get_bucket(bucket, validate=False)
        all_files = [k.name for k in bucket.list()]
        return all_files

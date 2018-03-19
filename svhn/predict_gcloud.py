from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import json
import numpy as np
from predict import load_images

PROJECT='svhn-digits'
MODEL_NAME='svhn_digits'
MODEL_VERSION='v2'

def authenticate_cloud():
    """Authenticate against Google Cloud. Returns a valid cloud API Resource"""
    credentials = GoogleCredentials.get_application_default()
    api = discovery.build('ml', 'v1', credentials=credentials)
    return api

def predict(img_json):
    api = authenticate_cloud()
    parent = 'projects/%s/models/%s/versions/%s' % (PROJECT, MODEL_NAME, MODEL_VERSION)
    response = api.projects().predict(body=img_json, name=parent).execute()

    if not 'predictions' in response or len(response['predictions']) < 1 or not 'labels' in response['predictions'][0]:
        raise RuntimeError('No valid predictions found in response')

    probas_out = response['predictions'][0]['labels']
    probas = np.array(probas_out).argmax(axis=-1)
    return probas

def read_json(file):
    return json.load(open(file))

def convert_img_to_json(img_array):
    """Convert img array into JSON format as expected by Google Cloud ML"""
    img = {}
    img["image"] = img_array.tolist()
    data = {}
    data["instances"] = [img]
    return data

## Run script
if __name__ == "__main__":
    imgs = load_images()
    for idx, img in enumerate(imgs):
        img_json = convert_img_to_json(imgs[idx])
        probas = predict(img_json)
        print probas

import asyncio, io, glob, os, sys, time, uuid, requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFile
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType
import numpy as np
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

def showImage(img):
    arr = np.array(img, dtype=np.uint8)
    plt.figure()
    plt.imshow(arr)
    
    
def getConfig(name = 'config.txt'):
    filename = name
    configs = {}
    try:
        with open(filename, 'r') as f:
            # It's assumed our file contains a single line,
            # with our API key
            for line in f:
                res = line.strip().split(" = ")
                configs[res[0]] = res[1]
            return configs
    except FileNotFoundError:
        print("'%s' file not found" % filename)

def normalizeDevito(image, dev_img, points):
    dev_array = np.array(dev_img)
    im_array = np.array(image)
    
    baby_forehead_sample = im_array[points.top][int(points.left+points.width/2)]#np.median(im_array[points.top][points.left:points.left+points.width], axis=(0, 1))
    normalization_val = np.array(np.abs(dev_array[0][int(dev_array.shape[1]/2)] - baby_forehead_sample), dtype=np.int16)

    
    
    print(normalization_val)
    dev_array = dev_array.astype(np.int16)
    dev_array -= np.array(normalization_val, dtype=np.uint8)
    indices = np.where(dev_array > 0)
    dev_array[indices] = np.negative(dev_array[indices])
    dev_array = [255,255,255] + dev_array
    dev_img = Image.fromarray(dev_array.astype('uint8'), 'RGB')
        
def placeDevito(image, dev_img, fc):
    points = fc.face_rectangle
   
    #normalizeDevito(image, dev_img, points)
    
    dev_img = dev_img.resize((points.width, points.height))
    image.paste(dev_img, box=(points.left,points.top,points.left+points.width,points.top+points.height), mask=None) 
    
def processImage(url):
    single_face_image_url = url
    response = requests.get(single_face_image_url)
    img = Image.open(io.BytesIO(response.content))

    devito_file = "d_0.png"
    #response2 = requests.get(devito_url)
    devito_img = Image.open(devito_file)
    devito_img.show()
    exit()
    
    
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    detected_faces = face_client.face.detect_with_url(url=single_face_image_url, return_face_attributes=['age','gender'])
    if not detected_faces:
        print("No faces, baby or otherwise.")
        return

    draw = ImageDraw.Draw(img)
    minFace = detected_faces[0]
    for face in detected_faces:
        print(face.face_rectangle)
        if float(face.face_attributes.age) < BABY_THRESHOLD:
            placeDevito(img, devito_img, face)
    
    #points = minFace.face_rectangle
    #devito_img = devito_img.resize((points.width, points.height))
    #img.paste(devito_img, box=(points.left,points.top,points.left+points.width,points.top+points.height), mask=None)   

    showImage(img)
    
if __name__ == "__main__":

    configs = getConfig()
    KEY = configs['key']
    ENDPOINT = configs['endpoint']
    BABY_THRESHOLD = 10
    
    urls = sys.argv[1:]
    
    for url in urls:
        print(url)
        processImage(url)
    plt.show()
    
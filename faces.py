import asyncio, io, glob, os, sys, time, uuid, requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFile
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType
import numpy as np
import matplotlib.pyplot as plt
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

def showImage(img):
    print("show")
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
    dev_array = np.array(dev_img, dtype=np.uint8)
    im_array = np.array(image)
    
    baby_forehead_sample = im_array[points.top][int(points.left+points.width/2)]#np.median(im_array[points.top][points.left:points.left+points.width], axis=(0, 1))
    baby_forehead_sample = np.append(baby_forehead_sample, 0)
    baby_forehead_sample = baby_forehead_sample.astype(np.uint8)
    normalization_val = dev_array[1][int(dev_array.shape[1]/2)] - baby_forehead_sample
    normalization_val[3] = 0

    print(baby_forehead_sample)
    print(dev_array[1][int(dev_array.shape[1]/2)])
    print(normalization_val)
    print(dev_array[1][0])
    dev_array += normalization_val
    print(dev_array[1][0])
    #print(normalization_val)
    #dev_array = dev_array.astype(np.int16)
    #dev_array -= np.array(normalization_val, dtype=np.uint8)
    #indices = np.where(dev_array > 0)
    #dev_array[indices] = np.negative(dev_array[indices])
    #dev_array = [255,255,255] + dev_array
    dev_img = Image.fromarray(dev_array.astype('uint8'), 'RGBA')
    return dev_img
        
def placeDevito(image, dev_img, fc):
    points = fc.face_rectangle
   
    dev_img = normalizeDevito(image, dev_img, points)
    
    dev_img = dev_img.resize((points.width, points.height))
    image.paste(dev_img, box=(points.left,points.top,points.left+points.width,points.top+points.height), mask=dev_img) 

def cloneDevito(image, dev_img, fc):
    points = fc.face_rectangle
    image = np.array(image, dtype=np.uint8)
    dev_alpha = dev_img
    dev_img = np.array(dev_img, dtype=np.uint8)
    dev_img = dev_img[:,:,:3]
    dev_img = cv2.resize(dev_img, dsize=(points.width, points.height), interpolation=cv2.INTER_NEAREST)#dev_img = dev_img.resize((points.width, points.height))
    
    
    
    src_mask = 255 * np.ones(dev_img.shape, dev_img.dtype)
    print(image.shape)
    print(dev_img.shape)
    print((int(points.left + (points.width/2)),int(points.top + (points.height/2))))
    image = cv2.seamlessClone(dev_img, image, src_mask, (int(points.left + (points.width/2)),int(points.top + (points.height/2))), cv2.NORMAL_CLONE)
    return image
    
def processImage(url):
    single_face_image_url = url
    response = requests.get(single_face_image_url)
    img = Image.open(io.BytesIO(response.content))

    devito_file = "d_0.png"
    #response2 = requests.get(devito_url)
    devito_img = Image.open(devito_file)
    
    
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
            img = cloneDevito(img, devito_img, face)
            break
    
    #points = minFace.face_rectangle
    #devito_img = devito_img.resize((points.width, points.height))
    #img.paste(devito_img, box=(points.left,points.top,points.left+points.width,points.top+points.height), mask=None)   
    print("about to show")
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
    
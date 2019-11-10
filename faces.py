import asyncio, io, glob, os, sys, time, uuid, requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFile
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType

ImageFile.LOAD_TRUNCATED_IMAGES = True
    
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

def placeDevito(image, dev_img, fc):
    points = fc.face_rectangle
    dev_img = dev_img.resize((points.width, points.height))
    image.paste(dev_img, box=(points.left,points.top,points.left+points.width,points.top+points.height), mask=None) 
    
if __name__ == "__main__":

    configs = getConfig()
    KEY = configs['key']
    ENDPOINT = configs['endpoint']
    BABY_THRESHOLD = 10
    
    
    single_face_image_url = sys.argv[1]
    response = requests.get(single_face_image_url)
    img = Image.open(io.BytesIO(response.content))

    devito_file = "devito1.png"
    #response2 = requests.get(devito_url)
    devito_img = Image.open(devito_file)
    
    
    
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    detected_faces = face_client.face.detect_with_url(url=single_face_image_url, return_face_attributes=['age','gender'])
    if not detected_faces:
        print("No faces, baby or otherwise.")
        exit()

    draw = ImageDraw.Draw(img)
    minFace = detected_faces[0]
    for face in detected_faces:
        print(face.face_rectangle)
        if float(face.face_attributes.age) < BABY_THRESHOLD:
            placeDevito(img, devito_img, face)
    
    #points = minFace.face_rectangle
    #devito_img = devito_img.resize((points.width, points.height))
    #img.paste(devito_img, box=(points.left,points.top,points.left+points.width,points.top+points.height), mask=None)   

    img.show()
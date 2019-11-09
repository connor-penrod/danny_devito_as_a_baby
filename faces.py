import asyncio, io, glob, os, sys, time, uuid, requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType


def getRectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    bottom = left + rect.height
    right = top + rect.width
    print(((left, top), (bottom, right)))
    return ((left, top), (bottom, right))
def getRectangle2(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    bottom = left + rect.height
    right = top + rect.width
    print(((left, top), (bottom, right)))
    return (left, top, right, bottom)
    
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
    
if __name__ == "__main__":

    configs = getConfig()
    KEY = configs['key']
    ENDPOINT = configs['endpoint']
    
    
    single_face_image_url = sys.argv[1]
    single_image_name = os.path.basename(single_face_image_url)
    response = requests.get(single_face_image_url)
    img = Image.open(BytesIO(response.content))

    devito_file = "C:\\Users\\Conno\\Classes\\CS4650\\final\\devito1.png"
    #response2 = requests.get(devito_url)
    devito_img = Image.open(devito_file)
    
    
    
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    detected_faces = face_client.face.detect_with_url(url=single_face_image_url, return_face_attributes=['age','gender'])
    if not detected_faces:
        raise Exception('No face detected from image {}'.format(single_image_name))

    draw = ImageDraw.Draw(img)
    minFace = detected_faces[0]
    for face in detected_faces:
        print(face.face_rectangle)
        if face.face_attributes.age < minFace.face_attributes.age:
            minFace = face
    
    points = minFace.face_rectangle
    devito_img = devito_img.resize((points.width, points.height))
    img.paste(devito_img, box=(points.left,points.top,points.left+points.width,points.top+points.height), mask=None)   

    img.show()
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

#probably useless
ImageFile.LOAD_TRUNCATED_IMAGES = True

def showImage(img):
    #show image using matplotlib

    print("show")
    arr = np.array(img, dtype=np.uint8)
    plt.figure()
    plt.imshow(arr)
    
    
def getConfig(name = 'config.txt'):
    #grabs API key and endpoint data from config.txt file

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
        
def showPoints(img, face):
    #this simply illustrates the 27 facial points that Azure Face API returns
    
    '''
     :param pupil_left:
    :type pupil_left: ~azure.cognitiveservices.vision.face.models.Coordinate
    :param pupil_right:
    :type pupil_right: ~azure.cognitiveservices.vision.face.models.Coordinate
    :param nose_tip:
    :type nose_tip: ~azure.cognitiveservices.vision.face.models.Coordinate
    :param mouth_left:
    :type mouth_left: ~azure.cognitiveservices.vision.face.models.Coordinate
    :param mouth_right:
    :type mouth_right: ~azure.cognitiveservices.vision.face.models.Coordinate
    :param eyebrow_left_outer:
    :type eyebrow_left_outer:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param eyebrow_left_inner:
    :type eyebrow_left_inner:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param eye_left_outer:
    :type eye_left_outer:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param eye_left_top:
    :type eye_left_top: ~azure.cognitiveservices.vision.face.models.Coordinate
    :param eye_left_bottom:
    :type eye_left_bottom:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param eye_left_inner:
    :type eye_left_inner:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param eyebrow_right_inner:
    :type eyebrow_right_inner:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param eyebrow_right_outer:
    :type eyebrow_right_outer:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param eye_right_inner:
    :type eye_right_inner:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param eye_right_top:
    :type eye_right_top:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param eye_right_bottom:
    :type eye_right_bottom:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param eye_right_outer:
    :type eye_right_outer:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param nose_root_left:
    :type nose_root_left:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param nose_root_right:
    :type nose_root_right:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param nose_left_alar_top:
    :type nose_left_alar_top:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param nose_right_alar_top:
    :type nose_right_alar_top:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param nose_left_alar_out_tip:
    :type nose_left_alar_out_tip:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param nose_right_alar_out_tip:
    :type nose_right_alar_out_tip:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param upper_lip_top:
    :type upper_lip_top:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param upper_lip_bottom:
    :type upper_lip_bottom:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param under_lip_top:
    :type under_lip_top:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    :param under_lip_bottom:
    :type under_lip_bottom:
     ~azure.cognitiveservices.vision.face.models.Coordinate
    '''
    m = face.face_landmarks
    
    #all the 27 available face landmarks
    areas = [m.pupil_left,m.pupil_right,m.mouth_left,m.mouth_right,m.nose_tip,m.eyebrow_left_inner,m.eyebrow_left_outer,m.eyebrow_right_inner,
             m.eyebrow_right_outer, m.eye_left_bottom, m.eye_left_inner, m.eye_left_outer, m.eye_left_top, m.eye_right_bottom, m.eye_right_inner,
             m.eye_right_outer, m.eye_right_top, m.nose_root_left, m.nose_root_right, m.nose_left_alar_top, m.nose_right_alar_top, m.nose_left_alar_out_tip,
             m.nose_right_alar_out_tip, m.upper_lip_top, m.upper_lip_bottom, m.under_lip_bottom, m.under_lip_top]
    
    draw = ImageDraw.Draw(img)
    
    #draw dots at all points
    for area in areas:
        x = area.x
        y = area.y
        radius = 2
        draw.ellipse([x-radius,y-radius,x+radius,y+radius], fill='red')

def normalizeDevito(image, dev_img, points):
    #normalizes devito's face to better match baby's skin tone.
    #UNDER CONSTRUCTION

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
    #simply pastes the devito image over the baby's face

    #get baby's face coordinates
    points = fc.face_rectangle
   
    #optionally normalize devito's face first
    #dev_img = normalizeDevito(image, dev_img, points)
    
    #resize devito image to baby's face size
    dev_img = dev_img.resize((points.width, points.height))
    
    #paste devito over baby's face
    image.paste(dev_img, box=(points.left,points.top,points.left+points.width,points.top+points.height), mask=dev_img) 

def cloneDevito(image, dev_img, fc):
    #this function runs poisson image cloning on devito and the baby's face
    
    #get bounding box around baby's face
    points = fc.face_rectangle
    
    #convert PIL image to numpy array
    image = np.array(image, dtype=np.uint8)
    
    #save alpha channel for later
    dev_alpha = dev_img
    
    #convert devito image to numpy array, discard alpha channel, and resize to fit baby's face
    dev_img = np.array(dev_img, dtype=np.uint8)
    dev_img = dev_img[:,:,:3]
    dev_img = cv2.resize(dev_img, dsize=(points.width, points.height), interpolation=cv2.INTER_NEAREST)#dev_img = dev_img.resize((points.width, points.height))
    
    
    #create source mask that is the size of the baby's face
    src_mask = 255 * np.ones(dev_img.shape, dev_img.dtype)
    
    #run cv2.seamlessClone using the two images and the source mask
    image = cv2.seamlessClone(dev_img, image, src_mask, (int(points.left + (points.width/2)),int(points.top + (points.height/2))), cv2.NORMAL_CLONE)
    return image
    
def processImage(url):
    #processImage handles all processing on each face in a given URL

    #load in family photo
    single_face_image_url = url
    response = requests.get(single_face_image_url)
    img = Image.open(io.BytesIO(response.content))

    #load in Danny Devito image
    devito_file = "d_0.png"
    devito_img = Image.open(devito_file)
    
    #create a FaceClient object using Azure credentials
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    #collect face information using Azure
    detected_faces = face_client.face.detect_with_url(url=single_face_image_url, return_face_landmarks=True, return_face_attributes=['age','gender'])
    if not detected_faces:
        print("No faces, baby or otherwise.")
        return

    for face in detected_faces:
        if float(face.face_attributes.age) < BABY_THRESHOLD:
            #perform processing on each individual baby's face
            showPoints(img, face)
            #img = cloneDevito(img, devito_img, face)
    
    showImage(img)
    
if __name__ == "__main__":

    #grab API key and endpoint from config.txt file
    #example config.txt:
    #   key = 1234
    #   endpoint = www.endpoint.com 
    configs = getConfig()
    KEY = configs['key']
    ENDPOINT = configs['endpoint']
    
    #this specifies minimum estimated age for a face to be considered a baby's face
    BABY_THRESHOLD = 10
    
    #grab urls from command line arguments
    urls = sys.argv[1:]
    
    for url in urls:
        processImage(url)
        
    #show edited image at the end
    plt.show()
    
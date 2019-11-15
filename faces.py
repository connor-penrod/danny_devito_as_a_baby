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
import normalize
import inspect

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
        
def getLandmarks(face):
    #takes an Azure Face object, and returns a dictionary containing "coordinate_name":(x_coord,y_coord) pairs
    attributes = inspect.getmembers(face.face_landmarks, lambda a:not(inspect.isroutine(a)))
    attrs = [a for a in attributes if True in [b in a[0] for b in ["pupil","nose","lip","eye"]]]
    dict = {}
    for attr in attrs:
        dict[attr[0]] = (attr[1].x, attr[1].y)
        
    return dict 
    
    
    
def drawMask(coord):
    coord.append(coord[0]) #repeat the first point to create a 'closed loop'
    xs, ys = zip(*coord) #create lists of x and y values

    plt.figure()
    plt.plot(xs,ys) 
    plt.show()
    
def drawPoints(image, pts):
    #utility function that will draw dots using whatever points array you give it on the image you give it
    #its a GENERAL PURPOSE FUNCTION. you hear that brian? GENERAL purpose
    image = Image.fromarray(image, 'RGB')
    for pt in pts:
        ImageDraw.Draw(image).ellipse([pt[0]-2,pt[1]-2,pt[0]+2, pt[1]+2], fill='red')
    return np.array(image, dtype=np.uint8)


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

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    
    #print(np.matmul(np.array([5,6]),rot_mat))
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
    
    
    return result
        
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
    
def maskResize(mask, size):
    #resizes a mask matrix to preserve as much of the mask values as possible
    #does not add interpolated values to the matrix either

    new_mask = np.zeros(size)
    proportion_x = size[0]/mask.shape[1]
    proportion_y = size[1]/mask.shape[0]
    
    for coords in zip(np.where(mask == 1)[0], np.where(mask == 1)[1]):
        new_mask[int(coords[0]*proportion_y)][int(coords[1]*proportion_x)] = 1
    return new_mask
    
    
    

def cloneDevito(image, dev_img, fc, devito_face):
    #this function runs poisson image cloning on devito and the baby's face
    
    #get bounding box around baby's face
    points = fc.face_rectangle
    dev_points = devito_face.face_rectangle   
    
    #convert PIL image to numpy array
    image = np.array(image, dtype=np.uint8)
    
    #save alpha channel for later
    dev_alpha = dev_img
    
    #convert devito image to numpy array, discard alpha channel
    dev_img = np.array(dev_img, dtype=np.uint8)
    dev_img = dev_img[:,:,:3]
    
    #get devito's face landmark coordinates
    devito_landmarks_dict = getLandmarks(devito_face)
  
    
    
    #create landmark mask matrix
    landmark_mask = np.zeros(dev_img.shape[0:2])
    for key in devito_landmarks_dict:
        lm = devito_landmarks_dict[key]
        landmark_mask[int(lm[1])][int(lm[0])] = 1
        print(int(lm[1]),int(lm[0]))
    
    #crop devito image to contain only devito's face
    dev_img = dev_img[dev_points.top:dev_points.top+dev_points.height,dev_points.left:dev_points.left+dev_points.width,:]
    landmark_mask = landmark_mask[dev_points.top:dev_points.top+dev_points.height,dev_points.left:dev_points.left+dev_points.width]
        
    #resize devito image and mask to match baby face size
    dev_img = cv2.resize(dev_img, dsize=(points.width, points.height), interpolation=cv2.INTER_NEAREST)
    landmark_mask = maskResize(landmark_mask, (points.width,points.height))#landmark_mask = cv2.resize(landmark_mask, dsize=(points.width, points.height), interpolation=cv2.INTER_NEAREST)
    print(len(np.where(landmark_mask == 1)[0]),len(np.where(landmark_mask > 0)[0]))
    
    #match rotation of babys face
    rotation = -(fc.face_attributes.head_pose.roll)
    dev_img = rotateImage(dev_img, rotation)
    landmark_mask = rotateImage(landmark_mask, rotation)

    #move landmark matrix into larger full image matrix and position properly over devito's face
    landmark_mask_full = np.zeros(image.shape[0:2])
    landmark_mask_full[points.top:points.top+points.height, points.left:points.left+points.width] = landmark_mask
    
    #retrieve landmark coordinates from matrix
    vals1 = np.where(landmark_mask_full == 1)
    vals2 = np.where(landmark_mask == 1)
    landmark_coords_full = list(zip(vals1[1], vals1[0]))  
    landmark_coords_relative = list(zip(vals2[1], vals2[0]))
    
    
    #normalize skin tones between devito and baby
    dev_img = normalize.normalize(dev_img, image[points.top:points.top+points.height, points.left:points.left+points.width])
    

    tight_mask = ((np.min(np.argwhere(landmark_mask == 1)[:,0]), np.min(np.argwhere(landmark_mask == 1)[:,1])),
                  (np.max(np.argwhere(landmark_mask == 1)[:,0]), np.max(np.argwhere(landmark_mask == 1)[:,1])))
    
    #create source mask
    src_mask = np.zeros(dev_img.shape,dev_img.dtype)
    
    src_mask[tight_mask[0][0]:tight_mask[1][0]+60,tight_mask[0][1]:tight_mask[1][1]] = 255
    
    #run cv2.seamlessClone using the two images and the source mask
    baby_landmarks_dict = getLandmarks(fc)
    baby_center = tuple([int(x) for x in np.mean([baby_landmarks_dict[y] for y in baby_landmarks_dict], axis=(0))])
    image = cv2.seamlessClone(dev_img, image, src_mask, baby_center, cv2.NORMAL_CLONE)
    
    #display landmarks as dots
    #image = drawPoints(image, landmark_coords_full)
    
    return image
    
def processImage(url):
    #processImage handles all processing on each face in a given URL

    #load in family photo
    single_face_image_url = url
    response = requests.get(single_face_image_url)
    img = Image.open(io.BytesIO(response.content))

    #load in Danny Devito image
    #devito_file = "d_0.png"
    devito_file = 'devito_uncropped_rot.png'
    #devito_file = 'devito1.png'
    devito_img = Image.open(devito_file)
    devito_stream = open(devito_file, 'r+b')

    
    #create a FaceClient object using Azure credentials
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    #collect face information using Azure
    detected_faces = face_client.face.detect_with_url(url=single_face_image_url, return_face_landmarks=True, return_face_attributes=['age','gender','headPose'])
    devito_face = face_client.face.detect_with_stream(devito_stream, return_face_landmarks=True)

    if not devito_face:
        print("Couldnt detect DEVITO")
        return

    if not detected_faces:
        print("No faces, baby or otherwise.")
        return

    for face in detected_faces:
        if float(face.face_attributes.age) < BABY_THRESHOLD:
            #perform processing on each individual baby's face
            #showPoints(img, face)                   
            img = cloneDevito(img, devito_img, face, devito_face[0])
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
    
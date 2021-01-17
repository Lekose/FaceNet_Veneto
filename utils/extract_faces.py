from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN

def extract_single_face_facenet(file, size=(160,160)):
    # extract single face from given image
    image = Image.open(file)
    # convert to RGB if required
    image = image.convert('RGB')
    # convert to numpp array
    pixel_array = asarray(image)
    # create our detector, uses default weights
    detector = MTCNN()
    # detect face in the image
    result = detector.detect_faces(pixels)
    # extract the bounding box from face
    x1, y1, width, height = result[0]['box']
    x2, y2 = x1 + width, y1 + width
    # extract face
    face = pixel_array[y1:y2, x1:x2]
    
    image = Image.fromarray(face)
    image = image.resize(size)
    face_array = asarray(image)
    return face_array
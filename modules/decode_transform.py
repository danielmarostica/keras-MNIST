import base64
import skimage.io
from skimage.color import rgb2gray
from skimage.transform import resize

'''
This script defines a decoder/transformer class.

    decode() method receives as input a base64-encoded image and returns it as numpy ndarray.
    
    to_tensor() method receives an ndarray image and reshapes it as a rank one tensor (28x28).
'''

class Transformer():
    def __init__(self):
        pass

    # decodes base64 image
    def decode(self, base64_string):
        imgdata = base64.b64decode(base64_string)
        img = skimage.io.imread(imgdata, plugin='imageio')
        return img
    
    # transforms image
    def to_tensor(self, input_image):
        # RGB to grayscale
        grayscaled = rgb2gray(input_image)
        
        # resizes
        resized = resize(grayscaled, (28, 28))
        
        # to tensors
        tensor_image = resized.reshape(-1,28,28,1)
        
        return tensor_image

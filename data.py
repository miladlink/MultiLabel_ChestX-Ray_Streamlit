from PIL import Image
from torchvision import transforms


tfms = transforms.Compose ([
    transforms.Resize ((224,224)),
    transforms.ToTensor (),
    transforms.Normalize (mean = [0.485, 0.456, 0.406], 
                          std = [0.229, 0.224, 0.225])
])


def preprocess (file):
    img = Image.open (file).convert ('RGB')
    tensor_img = tfms (img)
    tensor_img = tensor_img.unsqueeze (0)
    
    return tensor_img
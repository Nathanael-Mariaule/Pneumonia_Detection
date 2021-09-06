import cv2 as cv
import numpy as np
import os




	
def preprocessing(filepath, img_size=(150,150)):
    """
    	Resize, turn to grayscale, normalize value the image located at filepath
    	:param str filepath: location of the image
    	:param Tuple[int] img_size: resize values
    	:return np.array: 2D array with values of the preprocessed image  
    """
    img = cv.imread(filepath)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    img = cv.resize(img, img_size, interpolation = cv.INTER_AREA)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)   
    img = img/255
    img = np.reshape(img, (1, img_size[0],img_size[1],1))
    return img


def predict(filepath, mariaunet):
    """
        Use the tensorflow model mariaunet to predict pneumonia on the image located at filepath.
        The prediction will be written on the image
        :param str filepath: location of the image
        :param tf.keras.models mariaunet: the tensorflow model
    """
    img = preprocessing(filepath)
    preds = mariaunet.predict(img)[0][0]
    if preds>0.5:
        write(filepath, f'Normal\n {round(preds*100, 2)}% of confidence', color=(0,255,0))
    else:
        write(filepath, f'Pneumonia\n {round(100-preds*100,2)}% of confidence', color=(0,0,255))
        
        
def write(filepath, text, color):
    """
        write text in color in a rectangle on the top-left of the image located at filepath
        :param str filepath: location of the image
        :param str text: text to be display on the image
        :param Tuple[int]: BGR value of the color
    """
    img = cv.imread(filepath)
    font_scale = 2.5
    font = cv.FONT_HERSHEY_PLAIN
    thickness = 3
    (text_width, text_height) = cv.getTextSize(text.split('\n')[-1], font, fontScale=font_scale, thickness=thickness)[0]
    text_height = 2*text_height+5
    text_offset_x = img.shape[1] -10 -text_width
    text_offset_y = 25
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width , text_offset_y + 3*text_height ))
    cv.rectangle(img, box_coords[0], box_coords[1], color)
    dy = text_height + 5
    for i, line in enumerate(text.split('\n')):
        cv.putText(img, line.strip(), (text_offset_x+5, text_offset_y+(i+1)*dy), font, fontScale=font_scale, color=color, thickness=thickness)
        
    cv.imwrite(filepath, img)
    


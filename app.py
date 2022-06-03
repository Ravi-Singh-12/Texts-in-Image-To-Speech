import pytesseract as pt
from PIL import Image
from gtts import gTTS
from pytesseract import image_to_string
pt.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import cv2

def image_to_sound(path_to_image):
    """
    Function for converting an  image to sound
    """
    try:
        loaded_image = Image.open(path_to_image)
        decoded_text = image_to_string(loaded_image)
        cleaned_text = " ".join(decoded_text.split("\n"))
        print(cleaned_text)
       # txt=cleaned_text.lower()
       # print(txt)
        sound = gTTS(cleaned_text, lang="en")
        sound.save("s.mp3")
        return True
    except Exception as bug:
        print("The bug thrown while excuting the code\n", bug)
        return


if __name__ == "__main__":
    image_to_sound("image.jpg")
    input()
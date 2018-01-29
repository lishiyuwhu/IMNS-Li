from PIL import Image

img = Image.open('stego_.jpg')

img.save('stego.jpg', 'jpeg', quality = 100)
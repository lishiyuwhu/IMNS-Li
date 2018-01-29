from PIL import Image  
import numpy as np  
import matplotlib.pyplot as plt  

img2 = Image.open("stego_.jpg")
'''
img = np.array(Image.open("stego_.jpg"))  

# 随机生成500个椒盐  
rows, cols, dims = img.shape  
for i in range(500):  
    x = np.random.randint(200, rows)  
    y = np.random.randint(200, cols)  
    img[x, y, :] = 255  

img2 = Image.fromarray(img)
'''
img2.save('stego.jpg','jpeg', quality  = 100)
https://code.google.com/archive/p/f5-steganography/

To embed a file called msg.txt in a picture called pic.jpg run the following: 

 java -jar f5.jar e -e msg.txt pic.jpg out.jpg

To extract a message from a picture call you call this:

 java -jar f5.jar x -e out.txt out.jpg

You can also give a password to protect the data or specify the image quality for the JPEG, e.g.: 

java -jar f5.jar e -e msg.txt -p mypasswd -q 70 in.jpg out.jpg 

java -jar f5.jar x -p mypasswd -e out.txt in.jpg
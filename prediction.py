import os

try:
    import cv2

    print("importing cv2 completed")

    import tensorflow as tf
    print("importing tensorflow completed")

    import numpy as np
    print("importing numpy completed")

    from keras.preprocessing import image
    print("importing keras completed")

    import time
    import serial
    import collections
    from time import sleep
    
except Exception as e:

    print(f"error {e}")

    os.system("pip install opencv-python tensorflow numpy keras serial")

    import cv2
    print("importing cv2 completed") 

    import tensorflow as tf
    print("importing tensorflow completed")

    import numpy as np
    print("importing numpy completed")

    from keras.preprocessing import image
    print("importing keras completed")

    import time
    import serial
    import collections
    from time import sleep

inpros = False

sr = True

lister =[]

if sr == True:

    sre = serial.Serial("/dev/ttyUSB0",9600)

    print("success at connecting microcontroller")

print("library imported")

camera = cv2.VideoCapture(2)

print("loading model")

model = tf.keras.models.load_model("biosep v2.h5")

print("model loaded")

d_l = 0

labels = {0: 'bio', 1: 'non bio', 2: 'no object detected'}

while True:
     
    try:
     
     if inpros != True:

        ret, img = camera.read()

        cv2.imshow("biosep",img)

        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        img = image.img_to_array(img, dtype=np.uint8)

        img = tf.keras.applications.resnet50.preprocess_input(img)

        img = np.array(img) / 255.0

        s = time.time()

        p = model.predict(img[np.newaxis, ...])

        prob = np.max(p[0], axis=-1) 

        predicted_class = labels[np.argmax(p[0], axis=-1)]

        l = time.time()

        f_t = l - s

        if len(lister) != 8:

            lister.append(predicted_class)

        if len(lister) == 8:

            most_val,count = collections.Counter(lister).most_common(1)[0]
            
            print(f"result : {most_val}")

            lister.clear()

            if sr== True and d_l > 1:

                d_l = 0
                
                if most_val == "bio":

                    print("bio degrade")

                    sre.write(b'0')

                    sleep(3)

                if most_val == "non bio":

                    print("non bio degradeable")

                    sre.write(b'1')

                    sleep(3)

            if d_l < 2:

                d_l+=1

     keyboard_input = cv2.waitKey(1)
 
     if keyboard_input == 27: 
         
         break
     
    except Exception as e:

        print(e)

cv2.destroyAllWindows()

# share link : https://drive.google.com/file/d/1tzOw2PSemlka3nNz11XdZpbFwgIQ2Q_7/view?usp=sharing

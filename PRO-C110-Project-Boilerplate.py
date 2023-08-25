# To Capture Frame
import cv2

# To process image array
import numpy as np


# import the tensorflow modules and load the model
import tensorflow



# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)
myModel=tf.keras.models.load_models("keras_models.h5")

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		
		
		#resize the frame
		resizeFrame=cv2.resize(frame,(224,224))
		# expand the dimensions
		resizeFrame=np.expand_dims(resizeFrame,axis=0)

		# normalize it before feeding to the model
		resizeFrame=resizeFrame/255
		# get predictions from the model
		prediction=myModel.predict(resizeFrame)
		rock=int(prediction[0][0]*100)
		paper=int(prediction[0][1]*100)
		scissor=int(prediction[0][2]*100)
		print("rock:",rock)
		print("paper",paper)
		print("scissor",scissor)
		
		
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()

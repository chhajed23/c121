import cv2
import time
import numpy as np

# Saving the output as output.avi
fourcc=cv2.VideoWriter_fourcc(*'XVID')
output_file=cv2.VideoWriter("Output.avi",fourcc,20.0,(640,480))
cap=cv2.VideoCapture(0)
time.sleep(3)
bg=0
for i in range(60):
    ret,bg=cap.read()

bg=np.flip(bg,axis=1)  #Flipping the background because the camera captures the image inverted
while(cap.isOpened()):
    ret,img=cap.read()
    if not ret:
        break
    img=np.flip(img,axis=1)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)   #converting the bgr color to hsv color
    # genereting the mask to detect the red color
    lower_red=np.array([0,120,50])
    upper_red=np.array([10,255,255])
    mask_1=cv2.inRange(hsv,lower_red,upper_red)   #lower mask between 0 and 10

    #upper mask between 170 and 180
    lower_red=np.array([170,120,50])
    upper_red=np.array([180,255,255])
    mask_2=cv2.inRange(hsv,lower_red,upper_red)

    #Joining both the masks
    mask_1=mask_1+mask_2
    mask_1=cv2.morphologyEx(mask_1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask_1=cv2.morphologyEx(mask_1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    mask_2=cv2.bitwise_not(mask_1)
    res_1=cv2.bitwise_and(img,img,mask=mask_2)
    res_2=cv2.bitwise_and(bg,bg,mask=mask_1)
    # generating the final output by mixing res_1 and res_2
    final_output=cv2.addWeighted(res_1,1,res_2,1,0)
    output_file.write(final_output)
    cv2.imshow("Magic",final_output)
    cv2.waitKey(1)
    
cap.release()
output_file.release()
cv2.destroyAllWindows()


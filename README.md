# LicensePlateNumberExtractionUsingOCR
license plate number extraction using OCR
#install tesseract-OCR in your pc first
##then we need to install libraries:
#step 1: install ocr engine
!sudo apt install tesseract-ocr 

#step 2: installation for accessing ocr, edge and contour detection , visualization 
!pip install pytesseract opencv_python_headless matplotlib

#step 3: upload the image
from google.colab import files
uploaded=files.upload()

#step 4:read the image
##will take the path of image uploaded
image_path=next(iter(uploaded))
##after reading it store it
image=cv2.imread(image_path)

#step 5:
##convert the image into grayscale:
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#step 6:
##Reduce the noise using bilateral filter
filtered=cv2.bilateralFilter(gray,11,17,17)

#step 7:
##Now detect edges, Canny function will do that, thresholds are 30 and 200 here
edge=cv2.Canny(filtered,30,200)

#step 8:
##Now detect contours, we will retreive countours
contours , _=cv2.findContours(edge.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
##after that sorting of contours will be done from largest to smallest and we are taking top 10 contours here
countours=sorted(contours,key=cv2.contourArea,reverse=True)[:10]

#step 9:
##find object that is same as license plate, typically it is a rectangle with 4 sides
license_plate_contour=None
##find in all contours:
for contour in contours:
  approx=cv2.approxPolyDP(contour,10,True)
##check that contours are rectangle or not
  if len(approx)==4:
##we are finding valid contours
    license_plate_contour=approx
    break

#step 10:
##now we have to proceed with that valid contour, we will draw it
if license_plate_contour is not None:
  cv2.drawContours(image,[license_plate_contour],-1,(0,255,0),3)

##create mask with black color
  mask=cv2.fillPoly(gray.copy(),[license_plate_contour],0)
##convert that mask into grayscale
  masked_image=cv2.bitwise_and(gray,gray,mask=cv2.bitwise_not(mask))
##now we have to crop the image (that rectangle will be calculated)
  x,y,w,h=cv2.boundingRect(license_plate_contour)
  license_plate=gray[y:y+h,x:x+w]
##For displaying the origional image
  plt.imshow(cv2.cvtColor(license_plate,cv2.COLOR_BGR2RGB))
  plt.title('Detected plate')
  plt.show()

#step 11:
##apply OCR
  text=pytesseract.image_to_string(license_plate,config='--psm 11')
##display OCR
  print(text)
##in case there was no license plate found in image
else:
  print("not found")




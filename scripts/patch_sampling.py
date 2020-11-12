import sys
from PIL import Image

imageObject  = Image.open(sys.argv[1])
x_slide = int(sys.argv[2])
y_slide = int(sys.argv[3])
#cartesian coordinate system with 0,0 in upper left of image
x,y=imageObject.size

count = 0
prefix = sys.argv[1].split('/')[-1:][:-4]
for right in range(0,x):
    for lower in range(0,y):
        if right % x_slide == 0:
            if lower % y_slide == 0:
                left = right-250
                upper = lower-250
                if left >= 0:
                    if upper >= 0:
                        count += 1
                        cropped = imageObject.crop((left,upper,right,lower))
                        cropped = cropped.save(prefix+'_'+str(count)+".jpg") 




from utils import utils
import function_detect_edge as ced
from function_detect_line import*
import cv2 as cv2

imgs = utils.load_data('imgs/geometry.jpg')
utils.visualize(imgs, 'gray')
detector = ced.cannyEdgeDetector(imgs, sigma=1, kernel_size=5, lowthreshold=0.05, highthreshold=0.15, weak_pixel=20)
imgs_final = detector.detect()
utils.visualize(imgs_final, 'gray')


accumulator, thetas, rhos = hough_line(imgs_final)
show_hough_line(imgs_final, accumulator, thetas, rhos)

test = np.uint8(imgs_final)
 

lines = my_hough(imgs_final, rho=1, theta=np.pi/180, threshold=100)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(imgs,(x1,y1),(x2,y2),(0,255,255),2)    
cv2.imwrite('Hough Lines.jpg', imgs)
cv2.waitKey(0)
cv2.destroyAllWindows()
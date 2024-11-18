# GenData.py
"""
Đọc và xử lý ảnh đầu vào:

Ảnh đầu vào training_chars.png chứa các ký tự mẫu (chữ và số) mà hệ thống sẽ dùng để học.
Ảnh này được chuyển thành ảnh xám (grayscale), làm mờ để giảm nhiễu, và sau đó được chuyển đổi sang ảnh nhị phân đen-trắng (binary) để dễ dàng tách các ký tự ra khỏi nền.
Xác định các ký tự:

Dùng OpenCV để tìm các đường viền (contours) trong ảnh. Mỗi đường viền tương ứng với một vùng trong ảnh, có thể là một ký tự.
Những vùng có diện tích đủ lớn (dựa vào giá trị MIN_CONTOUR_AREA) sẽ được coi là một ký tự tiềm năng và tiếp tục xử lý.
Gắn nhãn cho ký tự và lưu trữ dữ liệu huấn luyện:

Với mỗi ký tự được tách ra, chương trình sẽ hiện cửa sổ cho người dùng nhập ký tự tương ứng. Người dùng nhấn phím để xác định ký tự (nhãn) của vùng đó.
Ảnh của từng ký tự sau khi được tách và resize sẽ được lưu vào một mảng lớn, đồng thời nhãn của mỗi ký tự sẽ được lưu vào một danh sách nhãn.
Cuối cùng, các mảng này được lưu vào các file classifications.txt (chứa nhãn) và flattened_images.txt (chứa ảnh ký tự đã làm phẳng).
"""
import numpy as np
import cv2
import sys


# module level variables ##########################################################################
MIN_CONTOUR_AREA = 40


RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
def main():
    imgTrainingNumbers = cv2.imread("training.png")            # read in training numbers image
    #imgTrainingNumbers = cv2.resize(imgTrainingNumbers, dsize = None, fx = 0.5, fy = 0.5)
    
    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)          # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                        # blur

                                                        # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean

    cv2.imshow("imgThresh", imgThresh)      # show threshold image for reference

    imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    npaContours, hierarchy = cv2.findContours(imgThreshCopy,        # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,                 # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)           # compress horizontal, vertical, and diagonal segments and leave only their end points

                                # declare empty numpy array, we will use this to write to file later
                                # zero rows, enough cols to hold all image data
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
   

    intClassifications = []         # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end

                                    # possible chars we are interested in are digits 0 through 9, put these in list intValidChars
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')] #Là mã ascii của mấy chữ này

    for npaContour in npaContours:                          # for each contour
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:          # if contour is big enough to consider
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)         # get and break out bounding rect

                                                # draw rectangle around each contour as we ask user for input
            cv2.rectangle(imgTrainingNumbers,           # draw rectangle on original training image
                          (intX, intY),                 # upper left corner
                          (intX+intW,intY+intH),        # lower right corner
                          (0, 0, 255),                  # red
                          2)                            # thickness

            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]                                  # crop char out of threshold image
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # resize image, this will be more consistent for recognition and storage

            cv2.imshow("imgROI", imgROI)                    # show cropped out char for reference
            cv2.imshow("imgROIResized", imgROIResized)      # show resized image for reference
            
            cv2.imshow("training_numbers.png", imgTrainingNumbers)      # show training numbers image, this will now have red rectangles drawn on it

            intChar = cv2.waitKey(0)                     # get key press

            if intChar == 27:                   # if esc key was pressed
                sys.exit()                      # exit program
            elif intChar in intValidChars:      # else if the char is in the list of chars we are looking for . . .

                intClassifications.append(intChar)        # append classification char to integer list of chars (we will convert to float later before writing to file)
                #Là file chứa label của tất cả các ảnh mẫu, tổng cộng có 32 x 5 = 160 mẫu.
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)                    # add current flattened impage numpy array to list of flattened image numpy arrays
                
            # end if
        # end if
    # end for

    fltClassifications = np.array(intClassifications, np.float32)                   # convert classifications list of ints to numpy array of floats
    
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   # flatten numpy array of floats to 1d so we can write to file later

    print ("\n\ntraining complete !!\n")

    np.savetxt("classifications.txt", npaClassifications)           # write flattened images to file
    np.savetxt("flattened_images.txt", npaFlattenedImages)          #

    cv2.destroyAllWindows()             # remove windows from memory

    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if

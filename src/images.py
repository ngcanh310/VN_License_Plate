import math
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import Preprocess

# Các hằng số và tham số
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

Min_char = 0.01
Max_char = 0.09

RESIZED_IMAGE_WIDTH = 20    
RESIZED_IMAGE_HEIGHT = 30

# Khởi tạo biến đếm biển số
n = 1

# Tải mô hình KNN một lần khi khởi động ứng dụng
def load_knn_model():
    try:
        npaClassifications = np.loadtxt("../dataset/classifications.txt", np.float32)
        npaFlattenedImages = np.loadtxt("../dataset/flattened_images.txt", np.float32)
        npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))  # reshape numpy array to 1d
        kNearest = cv2.ml.KNearest_create()
        kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
        return kNearest
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không tải được mô hình KNN: {str(e)}")
        return None

kNearest = load_knn_model()
if kNearest is None:
    exit()

def preprocess_image(img):
    """ Tiền xử lý hình ảnh: chuyển đổi ảnh gốc sang ảnh grayscale và áp dụng adaptive threshold """
    imgGrayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGrayscale, (5, 5), 0)
    imgThresh = cv2.adaptiveThreshold(
        imgBlurred,
        250,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_THRESH_BLOCK_SIZE,
        ADAPTIVE_THRESH_WEIGHT
    )
    return imgGrayscale, imgThresh

def recognize_license_plate(img_path):
    global n
    try:
        # Đọc ảnh và thay đổi kích thước
        img = cv2.imread(img_path)
        if img is None:
            messagebox.showerror("Lỗi", "Không thể đọc tệp ảnh. Vui lòng chọn lại.")
            return
        img = cv2.resize(img, dsize=(1920, 1080))

        # Tiền xử lý ảnh
        imgGrayscaleplate, imgThreshplate = preprocess_image(img)
        canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Canny Edge
        kernel = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(canny_image, kernel, iterations=1)  # Dilation

        # Tìm contours và lọc biển số xe
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lấy 10 contours có diện tích lớn nhất

        screenCnt = []
        for c in contours:
            peri = cv2.arcLength(c, True)  # Tính chu vi
            approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
            if len(approx) == 4:
                screenCnt.append(approx)

        if not screenCnt:
            messagebox.showwarning("Kết quả", "Không phát hiện biển số xe trong ảnh!")
            return

        # Xử lý từng biển số được phát hiện
        for sc in screenCnt:
            cv2.drawContours(img, [sc], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe

            # Tìm góc nghiêng của biển số
            (x1, y1) = sc[0, 0]
            (x2, y2) = sc[1, 0]
            (x3, y3) = sc[2, 0]
            (x4, y4) = sc[3, 0]
            array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            array.sort(key=lambda x: x[1], reverse=True)  # Sắp xếp theo trục y từ lớn đến nhỏ
            (x1, y1) = array[0]
            (x2, y2) = array[1]
            doi = abs(y1 - y2)
            ke = abs(x1 - x2)
            angle = math.atan(doi / ke) * (180.0 / math.pi) if ke != 0 else 0

            # Cắt và căn chỉnh biển số
            mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
            cv2.drawContours(mask, [sc], 0, 255, -1)
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))

            roi = img[topx:bottomx, topy:bottomy]
            imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
            ptPlateCenter = ((bottomx - topx) / 2, (bottomy - topy) / 2)

            # Xoay ảnh để căn chỉnh biển số
            if x1 < x2:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
            else:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

            roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
            imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
            roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
            imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

            # Phân đoạn và nhận diện ký tự
            kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
            cont, _ = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)  # Vẽ contour các ký tự trong biển số

            # Lọc ký tự
            char_x_ind = {}
            char_x = []
            height, width, _ = roi.shape
            roiarea = height * width

            for ind, cnt in enumerate(cont):
                (x, y, w, h) = cv2.boundingRect(cnt)
                ratiochar = w / h
                char_area = w * h

                if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                    if x in char_x:  # Đảm bảo x không trùng nhau
                        x = x + 1
                    char_x.append(x)
                    char_x_ind[x] = ind

            char_x = sorted(char_x)
            strFinalString = ""
            first_line = ""
            second_line = ""

            for i in char_x:
                (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                imgROI = thre_mor[y:y + h, x:x + w]  # Cắt ký tự
                imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # Resize ảnh
                npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)).astype(np.float32)

                # Nhận diện ký tự bằng KNN
                _, npaResults, _, _ = kNearest.findNearest(npaROIResized, k=3)
                strCurrentChar = chr(int(npaResults[0][0]))  # Chuyển từ ASCII sang ký tự
                cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)

                if y < height / 3:  # Xác định dòng ký tự
                    first_line += strCurrentChar
                else:
                    second_line += strCurrentChar

            recognized_plate = first_line + " " + second_line
            print(f"\n License Plate {n} is: {recognized_plate}\n")
            messagebox.showinfo("Kết quả", f"Phát hiện biển số xe: {recognized_plate}")
            n += 1

        # Hiển thị ảnh đã xử lý trong GUI
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_display = Image.fromarray(img_display)
        img_display = ImageTk.PhotoImage(img_display.resize((600, 400), Image.ANTIALIAS))
        img_label.config(image=img_display)
        img_label.image = img_display

    except Exception as e:
        messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {str(e)}")

# Hàm chọn file
def select_file():
    global selected_file
    selected_file = filedialog.askopenfilename(
        title="Chọn ảnh",
        filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*"))
    )
    if selected_file:
        display_image(selected_file)

# Hiển thị ảnh đã chọn
def display_image(img_path):
    img = Image.open(img_path)
    img.thumbnail((600, 400))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

# Tạo giao diện
root = tk.Tk()
root.title("Nhận dạng biển số xe")
root.geometry("800x600")

# Nút chọn ảnh
btn_select = tk.Button(root, text="Chọn ảnh", command=select_file, font=("Arial", 14))
btn_select.pack(pady=10)

# Nút nhận dạng biển số
btn_recognize = tk.Button(root, text="Nhận dạng biển số", command=lambda: recognize_license_plate(selected_file), font=("Arial", 14))
btn_recognize.pack(pady=10)

# Nơi hiển thị ảnh
img_label = tk.Label(root)
img_label.pack(pady=20)

root.mainloop()

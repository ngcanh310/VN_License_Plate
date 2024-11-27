# How to run:

## image.py: cd src -> python image.py. Nhận diện tốt, độ chính xác cao nhưng không có giao diện, thay đổi ảnh thủ công

## images.py: cd src -> python images.py. Giao diện dễ dùng, thân thiện nhưng nhận diện kém, cần cải thiện thuật toán. Sau khi chọn nhận dạng ảnh, hình ảnh sẽ được vẽ contour và in ra biển số xe

# Data

## Thư mục này chứa dữ liệu thô (raw data), dữ liệu gốc thu thập được trước khi xử lý. Dữ liệu này bao gồm ảnh, video

# Dataset

## Chứa dữ liệu đã qua xử lý và được tổ chức theo một cách có cấu trúc để sử dụng cho mục đích huấn luyện mô hình, phân tích dữ liệu

### training.png: ta viết các chữ số và kí tự (trừ kí tự O, I, J) với phông chữ “Biển số xe Việt Nam”, có thể xoay các kí tự này lần lượt với các góc -5°,5°,-10°,10

### classifications.txt: Dữ liệu phân loại (labels) của các ký tự trên biển số xe. Đây là thông tin về các nhãn mà mô hình học máy sẽ sử dụng để phân loại các ký tự (ví dụ: A, B, C, ...). Đây là phần quan trọng của dataset, vì nó chứa các nhãn tương ứng với mỗi ảnh ký tự đã được huấn luyện.

### flattened_images.txt: Dữ liệu hình ảnh đã được "làm phẳng" (flattened). Mỗi ảnh ký tự đã được biến đổi từ một ma trận 2D thành một vector 1D (mảng 1 chiều). Đây là đặc trưng của dataset, mà mô hình học máy sẽ sử dụng để học và phân loại.

# src

## Gendata.py: Đọc và xử lý ảnh đầu vào. Tạo các điểm dữ liệu KNN, bao gồm classifications.txt và flattened_images.txt.

## Preprocess.py: chứa các hàm sử lý hình ảnh, tạo ra ảnh xám và ảnh nhị phân từ ảnh gốc, nhằm làm nổi bật chi tiết của biển số xe, giúp nhận diện hoặc xử lý dễ dàng hơn

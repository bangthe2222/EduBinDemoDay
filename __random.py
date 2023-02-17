import os
import random
import shutil
def randomImgae():
    # Lấy danh sách tất cả các file ảnh trong thư mục "images"
    image_files = [f for f in os.listdir("images") if os.path.isfile(os.path.join("images", f))]

    # Nếu danh sách trống thì trả về None
    if not image_files:
        result = None
    else:
        # Chọn một file ảnh ngẫu nhiên
        image_file = random.choice(image_files)
        if not os.path.exists('used'):
            os.makedirs('used')
        # Di chuyển file ảnh sang thư mục "used"
        shutil.move(os.path.join("images", image_file), os.path.join("used", image_file))

        # Lấy hai chữ số cuối cùng của tên file trước dấu chấm
        result = int(image_file.split(".")[-2][-2:])

        # Trả về một số ngẫu nhiên từ 1 đến 3
        result = random.randint(1, 3)

    return result

import tkinter as tk
import tensorflow as tf
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np

root = Tk()
root.title('Object recognition for smart cities')
frame = tk.Frame(root, bg='#45aaf2')

lbl_pic_path = tk.Label(frame, text='Image Path:', padx=25, pady=25,
                        font=('verdana', 16), bg='#45aaf2')
lbl_result = tk.Label(frame, text='Đây là:', padx=25, pady=25,
                      font=('verdana', 16), bg='#45aaf2')
lbl_result1 = tk.Label(frame, text='', padx=25, pady=25,
                      font=('verdana', 16), bg='#45aaf2')
lbl_show_pic = tk.Label(frame, bg='#45aaf2')
entry_pic_path = tk.Entry(frame, font=('verdana', 16))
btn_browse = tk.Button(frame, text='CHOOSE', bg='grey', fg='#ffffff',
                       font=('verdana', 16))
detect = tk.Button(frame, text='Check', bg='blue', fg='#ffffff',
                   font=('verdana', 16))


def selectPic():
    global img
    resulf = {
        0.0: 'Không xác định', 1.0: 'Xe đạp', 2.0: 'Xe máy', 3.0: 'Xe ô tô', 4.0: 'Xe tải', 5.0: 'Xe khách',
        6.0: 'Xe ưu tiên', 7.0: 'Tàu điện', 8.0: 'Xe buýt', 9.0: 'Tàu thủy', 10.0: 'Máy bay thương mại',
        11.0: 'Máy bay quân sự', 12.0: 'Máy bay trực thăng', 13.0: 'Con người', 14.0: 'Con chó', 15.0: 'Con mèo',
        16.0: 'Nhà ở', 17.0: 'Vạch kẻ đường cho người đi bộ', 18.0: 'Đèn giao thông', 19.0: 'Đèn đường',
        20.0: 'Thùng rác công cộng', 21.0: 'Biển báo giao thông'
    }

    filename = filedialog.askopenfilename(
        initialdir="/images", title="Select Image",
        filetypes=( ("jpg images", "*.jpg"),("png images", "*.png"))
    )
    img = Image.open(filename)
    img = img.resize((200, 200), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    lbl_show_pic['image'] = img
    entry_pic_path.insert(0, filename)
    imgs = tf.keras.utils.load_img(filename, target_size=(128, 128))
    imgs = tf.keras.utils.img_to_array(imgs)
    imgs = np.expand_dims(imgs, axis=0)  # Add an extra dimension for batch
    imgs = imgs.astype('float32')
    imgs = imgs / 255.0
    model = tf.keras.models.load_model('D:\model_smart_city.h5')
    index = np.argmax(model.predict(imgs))
    print('Đây là:', resulf[index])
    lbl_result.configure( text ='Đây là: '+ resulf[index])
    

btn_browse['command'] = selectPic

frame.pack()

lbl_pic_path.grid(row=0, column=0)
entry_pic_path.grid(row=0, column=1, padx=(0, 20))
lbl_show_pic.grid(row=1, column=0, columnspan="2")
lbl_result.grid(row=2, column=0)
lbl_result1.grid(row=2, column=1)
btn_browse.grid(row=3, column=0, columnspan="2", padx=10, pady=10)
root.mainloop()
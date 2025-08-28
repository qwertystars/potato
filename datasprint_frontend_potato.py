import tkinter as tk
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk
#import matplotlib
#import 

window=tk.Tk()

def image_processor():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
    if file_path:
        new_window = tk.Toplevel(window)
        new_window.title("Image Viewer")
        new_window.geometry("1000x600")

        # Load and resize image
        img = Image.open(file_path)
        img_tk = ImageTk.PhotoImage(img)

        # Display image
        label = tk.Label(new_window, image=img_tk)
        label.image = img_tk   # keep reference
        label.pack(pady=20)


window.geometry("1000x550")

bg_pic=tk.PhotoImage(file="C:\\Users\\deept\\OneDrive\\Pictures\\Screenshots\\Screenshot 2025-08-28 132705.png")
label_bg=tk.Label(window,image=bg_pic)
label_bg.pack()

label1 = tk.Label(window, text="WANT TO KNOW YOUR PLANT CONDITION?", font=('Times new roman',30,'bold'),bg='#d9d9d9',fg='black')
label1.place(relx=0.1, rely=0.4)
label2 = tk.Label(window, text="SEND US YOUR PICTURE", font=('Times new roman',25,'bold'),bg='#d9d9d9',fg='black')
label2.place(relx=0.35, rely=0.5)

button = tk.Button(window, text="BROWSE PICTURE", font=("Arial",15),fg="black",bg="#00FFCC",relief='raised', borderwidth='5' ,command=image_processor)
button.place(relx=0.4,rely=0.6)


window.mainloop()

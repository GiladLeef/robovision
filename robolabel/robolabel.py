import argparse
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from segment_anything import sam_model_registry, SamPredictor
from utils.general import select_device
from utils.predict import SAM_setup, SAM_prediction
import torch
import time
import threading
import webbrowser

class ControlFrame(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)

        self.load_btn = tk.Button(self, text="Load Dataset", font='sans 10 bold', height=2, width=12, background="#343434", foreground="white", command=self.load_data)
        self.load_btn.pack(side=tk.LEFT, padx=(30, 30), pady=20, anchor="n")

        self.imageplayer = ttk.Frame(self)
        self.imagelabel = ttk.Label(self.imageplayer)
        self.imagelabel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        self.imageplayer.pack(side=tk.LEFT, padx=5, pady=5, expand=1, fill=tk.BOTH)

        self.side_tab = ttk.Frame(self)

        self.reset_btn = tk.Button(self.side_tab, text="Reset image", font='sans 10 bold', height=2, width=12, background="#343434", foreground="white", command=self.reset_annotation)
        self.reset_btn.pack(side=tk.BOTTOM, expand=1, padx=[10, 0], pady=[10, 10])

        self.new_btn = tk.Button(self.side_tab, text="New object", font='sans 10 bold', height=2, width=12, background="#343434", foreground="white", command=self.new_object)
        self.new_btn.pack(side=tk.BOTTOM, expand=1, padx=[10, 0], pady=[10, 10])

        self.scrollable_list = ttk.Frame(self.side_tab)
        self.scrollable_label = ttk.Label(self.scrollable_list, text="Object Labels", font='sans 10 bold')
        self.scrollable_label.pack(side=tk.TOP, expand=1, padx=[10, 0], pady=[10, 0], anchor="w")
        self.yScroll = tk.Scrollbar(self.scrollable_list, orient=tk.VERTICAL)
        self.yScroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.object_list = tk.Listbox(self.scrollable_list, height=10, width=20, background="white", foreground="#343434", activestyle='none')
        self.object_list.pack(side=tk.RIGHT, expand=1, padx=[10, 0], pady=[0, 10])
        self.object_list.config(yscrollcommand=self.yScroll.set)
        self.yScroll.config(command=self.object_list.yview)

        with open("classes.txt", 'r') as f:
            for line in f:
                self.object_list.insert('end', line.strip())

        self.scrollable_list.pack(side=tk.BOTTOM, expand=1, padx=[10, 0], pady=[10, 10])
        self.side_tab.pack(side=tk.LEFT, anchor="s")

        self.file_path = ""
        self.cur_annotation = []
        self.mask_images = []
        self.bbox_list = []
        self.annotation_count = None
        self.mask_count = None
        self.image_update_val = 100
        self.cur_image_index = 0
        self.image_list = []
        self.window_height = 0

        self.cur_image_path = os.path.join("assets", "bg.png")
        self.prev_image_path = ""

        self.predictor = SAM_setup("vit_h", os.path.join("model", "sam_vit_h_4b8939.pth"), select_device(""))
        self.frame_update()

        app.bind("<Right>", self.right_arrow_press)
        app.bind("<Left>", self.left_arrow_press)
        app.bind("<Control-z>", self.undo_annotation)
        app.bind("<n>", self.new_object)
        app.bind("<Control-Shift-Z>", self.undo_object)
        app.bind("<Control-s>", self.save_annotation)
        self.imagelabel.bind('<1>', self.left_key_press)
        self.imagelabel.bind('<3>', self.right_key_press)

        self.grid(column=0, row=0, padx=5, pady=5, sticky='ew')

    def load_data(self):
        self.file_path = filedialog.askdirectory()
        if self.file_path:
            image_extensions = [".jpg", ".png"]
            image_files = [x for x in os.listdir(self.file_path) if x.endswith(tuple(image_extensions))]
            if len(image_files) > 0:
                self.set_image(self.file_path)
            else:
                print("No images found in the selected directory")

    def set_image(self, file_path):
        self.file_path = file_path
        self.image_list = [x for x in os.listdir(self.file_path) if x.endswith(".jpg") or x.endswith(".png")]

        self.cur_image_path = os.path.join(self.file_path, self.image_list[0])
        self.cur_image_index = 0
        self.cur_annotation = []
        self.mask_images = []
        self.bbox_list = []

    def frame_update(self):
        self.window_height = app.winfo_height() - 20

        if self.cur_image_path != self.prev_image_path or len(self.cur_annotation) != self.annotation_count or self.window_height != self.prev_window_height or self.mask_count != len(self.mask_images):
            self.OCV_image = cv2.imread(self.cur_image_path)
            self.cv2image = cv2.cvtColor(self.OCV_image, cv2.COLOR_BGR2RGB)

            if self.cur_image_path != self.prev_image_path and self.prev_image_path != "":
                print("Loading the image. Please wait...")
                self.predictor.set_image(self.cv2image)
                print("Image loaded")

            self.prev_image_path = self.cur_image_path
            self.annotation_count = len(self.cur_annotation)
            self.prev_window_height = self.window_height
            self.mask_count = len(self.mask_images)

            self.img_height, self.img_width, _ = self.OCV_image.shape

            if len(self.cur_annotation) > 0:
                self.cv2image, self.mask_image, self.bbox_corners = SAM_prediction(self.cv2image, self.cur_annotation, self.predictor, self.img_height, self.img_width, self.mask_images)

                for i in range(len(self.cur_annotation)):
                    self.cv2image = cv2.circle(self.cv2image, (self.cur_annotation[i][0], self.cur_annotation[i][1]), int((self.img_height + self.img_width) / 200), self.cur_annotation[i][2], -1)

            img = Image.fromarray(self.cv2image)

            if int(self.img_width * self.window_height / self.img_height) > self.window_height:
                self.resized_image = img.resize((self.window_height, int(self.img_height * self.window_height / self.img_width)), Image.Resampling.LANCZOS)
                self.resize_type = "height"
                self.diff_dim = (self.window_height - self.resized_image.size[1]) / 2
            else:
                self.resized_image = img.resize((int(self.img_width * self.window_height / self.img_height), self.window_height), Image.Resampling.LANCZOS)
                self.resize_type = "width"
                self.diff_dim = (self.window_height - self.resized_image.size[0]) / 2

            image_new = Image.new("RGB", (self.window_height, self.window_height), (0, 0, 0))
            image_new.paste(self.resized_image, (int((self.window_height - self.resized_image.size[0]) / 2), int((self.window_height - self.resized_image.size[1]) / 2)))

            imgtk = ImageTk.PhotoImage(image_new)
            self.imagelabel.imgtk = imgtk
            self.imagelabel.configure(image=imgtk)

        self.imageplayer.after(self.image_update_val, self.frame_update)

    def right_arrow_press(self, event):
        if len(self.image_list) > 0:
            if self.cur_image_index < len(self.image_list) - 1:
                self.cur_image_index += 1
                self.cur_image_path = os.path.join(self.file_path, self.image_list[self.cur_image_index])
                self.cur_annotation = []
                self.mask_images = []
                self.bbox_list = []

    def left_arrow_press(self, event):
        if len(self.image_list) > 0:
            if self.cur_image_index > 0:
                self.cur_image_index -= 1
                self.cur_image_path = os.path.join(self.file_path, self.image_list[self.cur_image_index])
                self.cur_annotation = []
                self.mask_images = []
                self.bbox_list = []

    def left_key_press(self, event):
        if self.resize_type == "height":
            if event.y > (self.window_height - self.resized_image.size[1]) / 2 and event.y < self.window_height - (self.window_height - self.resized_image.size[1]) / 2:
                self.cur_annotation.append([int(event.x * self.img_width / self.window_height),
                                            int((event.y - self.diff_dim) * self.img_height / self.resized_image.size[1]), (0, 255, 0), 1])
        else:
            if event.x > (self.window_height - self.resized_image.size[0]) / 2 and event.x < self.window_height - (self.window_height - self.resized_image.size[0]) / 2:
                self.cur_annotation.append([int((event.x - self.diff_dim) * self.img_width / self.resized_image.size[0]),
                                            int(event.y * self.img_height / self.window_height), (0, 255, 0), 1])

    def right_key_press(self, event):
        if self.resize_type == "height":
            if event.y > (self.window_height - self.resized_image.size[1]) / 2 or event.y < self.window_height - (self.window_height - self.resized_image.size[1]) / 2:
                self.cur_annotation.append([int(event.x * self.img_width / self.window_height),
                                            int((event.y - self.diff_dim) * self.img_height / self.resized_image.size[1]), (255, 0, 0), 0])
        else:
            if event.x > (self.window_height - self.resized_image.size[0]) / 2 or event.x < self.window_height - (self.window_height - self.resized_image.size[0]) / 2:
                self.cur_annotation.append([int((event.x - self.diff_dim) * self.img_width / self.resized_image.size[0]),
                                            int(event.y * self.img_height / self.window_height), (255, 0, 0), 0])

    def reset_annotation(self):
        self.cur_annotation = []
        self.mask_images = []
        self.bbox_list = []

    def new_object(self, event=None):
        if len(self.cur_annotation) > 0:
            if len(self.object_list.curselection()) > 0:
                self.cur_annotation = []
                self.mask_images.append(self.mask_image)
                self.bbox_list.append([self.object_list.curselection()[0], *self.bbox_corners])
                self.object_list.selection_clear(0, tk.END)
                print("Previous masks:", len(self.mask_images))
            else:
                messagebox.showwarning("Warning", "Please select an object from the list")

    def undo_object(self, event):
        if len(self.mask_images) > 0:
            self.mask_images.pop()
            self.bbox_list.pop()
            print("Previous masks:", len(self.mask_images))

    def undo_annotation(self, event):
        if len(self.cur_annotation) > 0:
            self.cur_annotation.pop()

    def save_annotation(self, event):
        if len(self.mask_images) > 0:
            if len(self.cur_annotation) > 0 and len(self.object_list.curselection()) == 0:
                messagebox.showwarning("Warning", "Please select an object from the list before saving")
            else:
                if len(self.cur_annotation) > 0:
                    self.cur_annotation = []
                    self.mask_images.append(self.mask_image)
                    self.bbox_list.append([self.object_list.curselection()[0], *self.bbox_corners])
                    self.object_list.selection_clear(0, tk.END)

                with open(os.path.join(self.file_path, os.path.basename(self.cur_image_path).split(".")[0] + ".txt"), "w") as f:
                    for bbox in self.bbox_list:
                        f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")

        elif len(self.cur_annotation) > 0:
            if len(self.object_list.curselection()) > 0:
                with open(os.path.join(self.file_path, os.path.basename(self.cur_image_path).split(".")[0] + ".txt"), "w") as f:
                    f.write(f"{self.object_list.curselection()[0]} {self.bbox_corners[0]} {self.bbox_corners[1]} {self.bbox_corners[2]} {self.bbox_corners[3]}\n")
            else:
                messagebox.showwarning("Warning", "Please select an object from the list before saving")
        else:
            messagebox.showwarning("Warning", "Please label the image before saving")

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("robolabel tool")
        self.geometry("1120x650+10+10")
        self.minsize(1120, 650)
        self.robolabel_icon = ImageTk.PhotoImage(
            Image.open(os.path.join("assets", "bg.png")).resize((80, 80), Image.Resampling.LANCZOS))
        self.iconphoto(False, self.robolabel_icon)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.quit()
        self.destroy()
        os._exit(1)

if __name__ == "__main__":
    app = App()
    controllerframe = ControlFrame(app)
    app.mainloop()
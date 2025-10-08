import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import torch
from torchvision import transforms
from torchvision.transforms import Compose
from mnist import DigitModel
import torchvision.transforms.v2 as tfs


class MNISTTester:
    def __init__(self, root: str, transform: Compose | None = None):
        self.root = root
        self.root.title("MNIST Model Tester")
        self.model = None
        self.transform = transforms

        # Холст для рисования
        self.canvas_size = 280  # 10x масштаб для удобства рисования
        self.img_size = 28
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas.grid(row=0, column=0, rowspan=4)

        self.image = Image.new("L", (self.img_size, self.img_size), 'white')
        self.draw = ImageDraw.Draw(self.image)
        

        self.last_x, self.last_y = None, None
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset_pos)

        # Кнопки
        tk.Button(root, text="Загрузить модель", command=self.load_model).grid(row=0, column=1, sticky='ew')
        tk.Button(root, text="Очистить", command=self.clear_canvas).grid(row=1, column=1, sticky='ew')
        tk.Button(root, text="Predict", command=self.predict).grid(row=2, column=1, sticky='ew')

        # Для графика
        self.figure = plt.Figure(figsize=(4,3))
        self.ax = self.figure.add_subplot(111)
        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_plot.get_tk_widget().grid(row=0, column=2, rowspan=4)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            # Рисуем линию на холсте
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=20, fill='white', capstyle=tk.ROUND, smooth=True)
            # Рисуем на изображении (масштабируем координаты)
            x1, y1 = self.last_x * self.img_size // self.canvas_size, self.last_y * self.img_size // self.canvas_size
            x2, y2 = x * self.img_size // self.canvas_size, y * self.img_size // self.canvas_size
            self.draw.line([x1, y1, x2, y2], fill='black', width=2)
        self.last_x, self.last_y = x, y

    def reset_pos(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.img_size, self.img_size), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.ax.clear()
        self.canvas_plot.draw()

    def load_model(self):
        path = filedialog.askopenfilename(title="Выберите файл модели")
        if path:
            # Для PyTorch:
            self.model_data = torch.load(path, map_location=torch.device('cpu'))
            self.model = DigitModel()
            self.model.load_state_dict(self.model_data)
            self.model.eval()
            tk.messagebox.showinfo("Модель", "Модель успешно загружена!")

    def predict(self):
        if self.model is None:
            tk.messagebox.showwarning("Модель", "Сначала загрузите модель!")
            return

        # Берём копию изображения с холста
        img = self.image.copy()

        # MNIST — чёрный фон, белая цифра → инвертируем
        img = ImageOps.invert(img)

        # Преобразование как на валидации
        transform = tfs.Compose([
            tfs.ToImage(),
            tfs.ToDtype(torch.float32, scale=True),
            tfs.Normalize((0.5,), (0.5,))
        ])

        img = transform(img).unsqueeze(0)   # [1, 1, 28, 28]

        with torch.no_grad():
            output = self.model(img)
            probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

        # Отладка — посмотреть, что реально подаём в сеть
        import matplotlib.pyplot as plt
        plt.imshow(img.squeeze().cpu().numpy(), cmap="gray")
        plt.title("Вход в модель")
        plt.show()

        # Визуализация распределения классов
        self.ax.clear()
        self.ax.bar(range(10), probs)
        self.ax.set_xticks(range(10))
        self.ax.set_xlabel("Класс")
        self.ax.set_ylabel("Уверенность")
        self.ax.set_title("Распределение по классам")
        self.canvas_plot.draw()

if __name__ == "__main__":
    transforms = None
    
    root = tk.Tk()
    app = MNISTTester(root, transforms)
    root.mainloop()

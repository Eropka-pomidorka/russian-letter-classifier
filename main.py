import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
import PIL.Image
import PIL.ImageDraw
import numpy as np
import network as net
import DB_loader as dbl

class DrawingApp:
    def __init__(self, master):
        self.master = master
        master.title("Распознавание рукописных букв")

        self.nn = None
        self.image_size = 28
        self.eraser_size = tk.IntVar(value=10)
        self.image = PIL.Image.new("L", (200, 200), "white")
        self.draw = PIL.ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None
        self.drawing_mode = "pen"  # Начальный режим рисования - ручка


        self.create_start_widgets()  # Создаем виджеты стартового окна

    def create_start_widgets(self):
        self.start_frame = tk.Frame(self.master)
        self.start_frame.pack(expand=True)  # Разрешаем фрейму расширяться

        self.title_label = tk.Label(self.start_frame, text="Распознавание\nрукописных букв",
                                     font=("Arial", 24))  # Измените текст при необходимости
        self.title_label.pack(pady=20)

        button_frame = tk.Frame(self.start_frame)  # Фрейм для кнопок
        button_frame.pack()

        self.load_model_button = tk.Button(button_frame, text="Загрузить модель", command=self.load_model)
        self.load_model_button.pack(side="left", padx=10, pady=10)

        self.train_model_button = tk.Button(button_frame, text="Обучить модель",
                                           command=self.create_and_train_model)
        self.train_model_button.pack(side="left", padx=10, pady=10)

        # Настраиваем масштабирование кнопок
        self.start_frame.rowconfigure(0, weight=1)  # Строка 0 (метка) расширяется
        self.start_frame.columnconfigure(0, weight=1)  # Столбец 0 (фрейм с кнопками) расширяется
        button_frame.columnconfigure(0, weight=1)  # Кнопки расширяются по горизонтали
        button_frame.columnconfigure(1, weight=1)

        self.master.update_idletasks()  # Обновляем геометрию окна
        min_width = self.master.winfo_width()
        min_height = self.master.winfo_height()
        self.master.minsize(min_width, min_height)

    def create_drawing_widgets(self):
        self.start_frame.destroy()  # Удаляем стартовое окно

        # Создаем виджеты для рисования
        canvas_frame = tk.Frame(self.master)
        canvas_frame.grid(row=0, column=0)

        self.canvas = tk.Canvas(canvas_frame, width=200, height=200, bg='white')
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_point)
        self.canvas.pack(side="left")

        self.clear_button = tk.Button(canvas_frame, text="Очистить", command=self.clear_canvas)
        self.clear_button.pack(side="left")

        result_frame = tk.Frame(self.master)
        result_frame.grid(row=1, column=0)

        self.recognize_button = tk.Button(result_frame, text="Распознать", command=self.recognize_letter)
        self.recognize_button.pack(side="left")

        self.result_label = tk.Label(result_frame, text="")
        self.result_label.pack(side="left")

        #self.mode_button = tk.Button(canvas_frame, text="Ластик", command=self.toggle_drawing_mode)
        #self.mode_button.pack(side="left")

        # # Создаем фрейм для ползунка с отступами
        # slider_frame = tk.Frame(canvas_frame)
        # slider_frame.pack(side="left", padx=10, pady=5)  # Отступы для фрейма

        self.eraser_size_label = tk.Label(canvas_frame, text="Размер ластика:")
        self.eraser_size_label.pack(side="left")



        self.eraser_size_slider = ttk.Scale(
        canvas_frame, from_=1, to=50, orient=tk.HORIZONTAL,
        command=self.update_eraser_size, variable=self.eraser_size)
        
        self.eraser_size_slider.pack(side="left")

        # Кнопка для переключения режима рисования (после ползунка)
        self.mode_button = tk.Button(canvas_frame, text="Ластик", command=self.toggle_drawing_mode)
        self.mode_button.pack(side="left")

        self.eraser_circle = None  # Инициализируем ID круга ластика
        self.canvas.bind("<Motion>", self.update_eraser_preview)
        self.canvas.bind("<Leave>", self.hide_eraser_preview)

    def draw_line(self, event):
        x, y = event.x, event.y
        r = self.eraser_size.get() // 2  # Радиус ластика

        if self.last_x is not None and self.last_y is not None:
            if self.drawing_mode == "pen":
                self.canvas.create_line(self.last_x, self.last_y, x, y, width=10, capstyle=tk.ROUND, smooth=True)
                self.draw.line([self.last_x, self.last_y, x, y], fill="black", width=10)
            elif self.drawing_mode == "eraser":
                self.canvas.create_oval(x-r, y-r, x+r, y+r, outline="white", fill="white")
                self.draw.ellipse([x-r, y-r, x+r, y+r], outline="white", fill="white")
        self.last_x, self.last_y = x, y

    def toggle_drawing_mode(self):
        if self.drawing_mode == "pen":
            self.drawing_mode = "eraser"
            self.mode_button["text"] = "Ручка"
        else:
            self.drawing_mode = "pen"
            self.mode_button["text"] = "Ластик"

    def update_eraser_size(self, value):
        # Эта функция больше не нужна для обновления размера ластика, 
        # но она должна быть определена, так как используется в self.eraser_size_slider
        pass 

    def update_eraser_preview(self, event):
        x, y = event.x, event.y
        r = self.eraser_size.get() // 2

        if self.drawing_mode == "eraser":
            if self.eraser_circle:
                self.canvas.delete(self.eraser_circle)
            self.eraser_circle = self.canvas.create_oval(x-r, y-r, x+r, y+r, outline="black", width=1)
        
    def hide_eraser_preview(self, event):
        if self.eraser_circle:
            self.canvas.delete(self.eraser_circle)
            self.eraser_circle = None

    def reset_last_point(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = PIL.Image.new("L", (200, 200), "white")
        self.draw = PIL.ImageDraw.Draw(self.image)

    def load_model(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.nn = net.load(file_path)
            self.result_label["text"] = "Модель загружена"
            self.create_drawing_widgets()  # Создаем виджеты для рисования

    def create_and_train_model(self):
        try:
            layers = simpledialog.askstring("Слои", "Введите количество нейронов в каждом слое, разделенных запятыми (например, 784,30,33):")
            layers = [int(x.strip()) for x in layers.split(",")]
            if len(layers) < 2 or layers[0] != 784:
                raise ValueError("Неверный формат слоев. Первый слой должен иметь 784 нейрона.")

            self.nn = net.Network(layers)

            epochs = simpledialog.askinteger("Эпохи", "Введите количество эпох:")
            mini_batch_size = simpledialog.askinteger("Размер мини-пакета", "Введите размер мини-пакета:")
            eta = simpledialog.askfloat("Скорость обучения", "Введите скорость обучения:")
            lmbda = simpledialog.askfloat("Lmbda", "Введите значение Lmbda (регуляризация):")

            training_data, _, test_data = dbl.load_data_wrapper()
            self.nn.SGD(training_data, epochs, mini_batch_size, eta, lmbda=lmbda, test_data=test_data)

            self.result_label["text"] = "Модель обучена"
            self.create_drawing_widgets()  # Создаем виджеты для рисования

        except (TypeError, ValueError) as e:
            messagebox.showerror("Ошибка", str(e))

    def recognize_letter(self):
        if self.nn is None:
            messagebox.showwarning("Предупреждение", "Сначала создайте или загрузите модель.")
            return

        image = self.image.resize((self.image_size, self.image_size), PIL.Image.Resampling.LANCZOS)
        input_data = np.array(image).reshape(self.image_size * self.image_size, 1) / 255.0

        output = self.nn.feedforward(input_data)
        predicted_class = np.argmax(output)

        #letter = chr(ord('a') + predicted_class)
        letter = net.letter(predicted_class)
        self.result_label["text"] = f"Распознанная буква: {letter}"

root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
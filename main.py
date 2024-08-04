import flet as ft
from PIL import Image
import flet.canvas as cv
import PIL.ImageDraw
import numpy as np
import network as net
import DB_loader as dbl

class State:
    x: float
    y: float

state = State()

def main(page: ft.Page):
    page.title = "Классификатор"
    page.theme_mode = "dark"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    page.model = None
    page.letter = None
    page.eraser = False

    def load_model(e: ft.FilePickerResultEvent):
        if e.files:
            inf_field.value = ''
            path = ''
            for file in e.files:
                path = file.path
                page.model = net.load(path)
                inf_field.value = "Загружена модель <{}>".format(file.name)
            page.open(dlg_modal)
        else:
            inf_field.value = 'Модель не выбрана'

        page.update()

    pick_dialog = ft.FilePicker(on_result=load_model)
    page.overlay.append(pick_dialog)
    inf_field = ft.Text()


    def activate_creation_field():
        layers_field.visible = True
        creation_confirm_btn.visible = True
        page.update()

    def create_model():
        layers_str = layers_field.value

        layers = []

        layers.append(784)
        for hidden_layer in layers_str.split(","):
            layers.append(int(hidden_layer))
        layers.append(33)
            
        page.model = net.Network(layers)

        inf_field.value = f"Текущая модель имеет {layers} нейронов с слоях"        

        epochs_field = ft.TextField(label="Количество эпох", value="300", keyboard_type=ft.KeyboardType.NUMBER)
        batch_size_field = ft.TextField(label="Размер мини-пакета", value="15", keyboard_type=ft.KeyboardType.NUMBER)
        eta_field = ft.TextField(label="Скорость обучения", value="0.01", keyboard_type=ft.KeyboardType.NUMBER)
        lmbda_field = ft.TextField(label="Параметр регуляризации L2", value="22.0", keyboard_type=ft.KeyboardType.NUMBER)

        def start_training():
            layers_field.visible = False
            creation_confirm_btn.visible = False
            page.update()

            epochs = int(epochs_field.value)
            mini_batch_size = int(batch_size_field.value)
            eta = float(eta_field.value)
            lmbda = float(lmbda_field.value)

            page.close(dlg_hyperparams)

            def update_progress(epoch):
                learning_progress_bar.value = epoch / epochs
                title_pb.value = f"Обучение завершено на: {learning_progress_bar.value * 100:.0f}%"
                page.update()

            title_pb.visible = True
            learning_progress_bar.visible = True
            page.update()

            title_pb.value = "Загрузка датасета..."
            page.update()
            training_data, test_data = dbl.load_dataset()
            page.model.SGD(training_data, epochs, mini_batch_size, eta, lmbda, test_data, callback=update_progress)
            
            title_pb.value = "Обучение завершено!"
            learning_progress_bar.visible=False
            page.update()
            page.model.save()

            if len(page.navigation_bar.destinations) == 1:
                page.navigation_bar.destinations.append(ft.NavigationBarDestination(
                    icon=ft.icons.DRAW, 
                    label="Inference"))
            page.update()
        
            page.navigation_bar.selected_index = 1
            navigate(None)
            page.update()

        dlg_hyperparams = ft.AlertDialog(
            modal=True,
            title=ft.Text("Введите значения гиперпараметров"),
            content=ft.Column(
                [
                    epochs_field,
                    batch_size_field,
                    eta_field,
                    lmbda_field,
                ]
            ),
            actions=[
                ft.TextButton("Отмена", on_click=lambda e: page.close(dlg_hyperparams)),
                ft.ElevatedButton("Начать обучение", on_click=lambda _: start_training()),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        page.open(dlg_hyperparams)
        page.update()        

    def handle_close(e):
        page.close(dlg_modal)
        page.update()
    
    def go_to_next_step(e):
        page.close(dlg_modal)
        page.update()

        if len(page.navigation_bar.destinations) == 1:
            page.navigation_bar.destinations.append(ft.NavigationBarDestination(
                icon=ft.icons.DRAW, 
                label="Inference"))
            page.update()
        page.navigation_bar.selected_index = 1
        navigate(None)
        page.update()


    dlg_modal = ft.AlertDialog(
        modal=True,
        title=ft.Text("Модель готова к работе!"),
        content=ft.Text("Хотите перейти к следующему этапу?"),
        actions=[
            ft.TextButton("Да", on_click=go_to_next_step),
            ft.TextButton("Нет", on_click=handle_close),
        ],
        on_dismiss=handle_close,
        actions_alignment=ft.MainAxisAlignment.END
    )

    layers_field = ft.TextField(label="Задайте количество нейронов в скрытом/скрытых слое/слоях. Например: 256, 128", 
                                value="256, 128", keyboard_type=ft.KeyboardType.NUMBER,
                                visible=False)

    creation_confirm_btn = ft.ElevatedButton("Подтвердить",
                                             on_click=lambda _: create_model(),
                                             visible=False)

    title_pb = ft.Text("", visible=False)

    learning_progress_bar = ft.ProgressBar(width=400, visible=False)

    preparing_page = ft.Column(
        [
            ft.Row([ft.Text("Классификация\nрукописных букв\nрусского алфавита", size=42, text_align=ft.TextAlign.CENTER)], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row(
                [
                    ft.ElevatedButton("Загрузить модель",
                                        icon=ft.icons.UPLOAD_FILE,
                                        on_click=lambda _: pick_dialog.pick_files()),
                    ft.ElevatedButton("Создать модель", 
                                        icon=ft.icons.CREATE,
                                        on_click=lambda _: activate_creation_field())
                ], alignment=ft.MainAxisAlignment.CENTER, height=40
            ),
            ft.Row([inf_field], height=50, alignment=ft.MainAxisAlignment.CENTER),
            ft.Row([layers_field, creation_confirm_btn], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row(
                [
                    ft.Column(
                        [
                            title_pb,
                            learning_progress_bar
                        ]
                    )
                ], alignment=ft.MainAxisAlignment.CENTER
            )
        ], spacing=20
    )

    def pan_start(e: ft.DragStartEvent):
        state.x = e.local_x
        state.y = e.local_y

    def pan_update(e: ft.DragUpdateEvent):
        circle_radius = 15 if page.eraser else 10
        color = "white" if page.eraser else "black"
        canvas_letter.shapes.append(
            cv.Circle(
                e.local_x, e.local_y, circle_radius, paint=ft.Paint(color=color, stroke_width=3), data=color
            )
        )
        canvas_letter.update()
        state.x = e.local_x
        state.y = e.local_y

    def toggle_eraser(e):
        page.eraser = not page.eraser
        if page.eraser:
            tool_button.text="Ластик"
        else:
            tool_button.text="Кисть"
        page.update()

    def clear_canvas(e):
        canvas_letter.shapes = canvas_letter.shapes[:1]
        canvas_letter.update()

    def recognize_drawing(e):
        image = get_canvas_image(canvas_letter)
        #image.save("canvas_image.png")

        input_data = process_image_for_network(image)
        output = page.model.feedforward(input_data)

        # Находим индексы трех максимальных вероятностей
        top_indices = np.argsort(output, axis=0)[-3:].flatten()[::-1]

        top_probabilities = output[top_indices] * 100
        other_probability = 100 - np.sum(top_probabilities)  

        message = "Три наиболее вероятные буквы:\n"
        for i, prob in zip(top_indices, top_probabilities):
            message += f"{net.letter(i)}: {prob[0]:.2f}%\n"
        message += f"Остальное: {other_probability:.2f}%"

        page.snack_bar = ft.SnackBar(ft.Text(message))
        page.snack_bar.open = True
        page.update()

    def get_canvas_image(canvas):
        image = PIL.Image.new("RGB", (canvas_size, canvas_size), "white")
        draw = PIL.ImageDraw.Draw(image)

        for control in canvas.shapes:
            if isinstance(control, cv.Circle):
                x = control.x
                y = control.y
                radius = control.radius
                color = control.data

                # Рисуем круг на PIL изображении
                draw.ellipse(
                    [(x - radius, y - radius), (x + radius, y + radius)],
                    fill=color, 
                    outline=color
                )
        return image

    def process_image_for_network(image):
        image = image.convert("L")
        image = image.resize((28, 28), PIL.Image.Resampling.LANCZOS)
        image.save("canvas_image_28.png")
        image_array = np.array(image) / 255.0
        input_data = image_array.reshape(784, 1)
        return input_data

    def load_letter(e: ft.FilePickerResultEvent):
        if e.files:
            for file in e.files:
                input_image = Image.open(file.path).convert('L')
                input_data = process_image_for_network(input_image)
                output = page.model.feedforward(input_data)

                top_indices = np.argsort(output, axis=0)[-3:].flatten()[::-1]

                top_probabilities = output[top_indices] * 100
                other_probability = 100 - np.sum(top_probabilities)

                message = "Три наиболее вероятные буквы:\n"
                for i, prob in zip(top_indices, top_probabilities):
                    message += f"{net.letter(i)}: {prob[0]:.2f}%\n"
                message += f"Остальное: {other_probability:.2f}%"

                page.snack_bar = ft.SnackBar(ft.Text(message))
                page.snack_bar.open = True

        page.update()

    pick_letter = ft.FilePicker(on_result=load_letter)
    page.overlay.append(pick_letter)

    canvas_size = 300
    canvas_letter = cv.Canvas(
        shapes=[cv.Fill(ft.Paint(color="white"))],
        width=canvas_size,
        height=canvas_size,
        content=ft.GestureDetector(
            on_pan_start=pan_start,
            on_pan_update=pan_update,
            drag_interval=10,
        ),
        expand=False,
    )

    tool_button = ft.ElevatedButton(text="Кисть", on_click=toggle_eraser)

    inference_page = ft.Column(
        [
            ft.Row([ft.Text("Нарисуйте рукописную букву русского алфавита")], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row(
                [
                    ft.Container
                    (
                        canvas_letter,
                        border_radius=5,
                        width=canvas_size,
                        height=canvas_size,
                        expand=False,
                    )
                ], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row(
                [
                    ft.ElevatedButton(text="Распознать",
                                      icon=ft.icons.REMOVE_RED_EYE_OUTLINED,
                                      on_click=recognize_drawing),
                    ft.ElevatedButton(text="Очистить",
                                      icon=ft.icons.CLEAR,
                                      on_click=clear_canvas),
                                      tool_button
                ], alignment=ft.MainAxisAlignment.CENTER, height=40),
            ft.Row(
                [
                    ft.Text("""Для большей точности рекомендуется, рисовать по центру.
                             \nБуквы также лучше рисовать среднего размера""",size=11)
                ], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row(
                [
                    ft.Text("Также можно"),
                    ft.TextButton(text="загрузить изображение буквы",
                                  on_click=lambda _: pick_letter.pick_files(allowed_extensions=["png", "jpeg", "jpg"]))
                ], alignment=ft.MainAxisAlignment.CENTER, height=40
            )
        ], spacing=20
    )

    def navigate(e):
        index = page.navigation_bar.selected_index
        page.clean()

        if index == 0: page.add(preparing_page)
        elif index == 1: page.add(inference_page)

    page.navigation_bar = ft.NavigationBar(
        destinations=[
            ft.NavigationBarDestination(icon=ft.icons.HOME, label="Подготовка модели")
        ], on_change=navigate
    )

    page.add(preparing_page)

#ft.app(target=main, view=ft.AppView.WEB_BROWSER)
ft.app(target=main)
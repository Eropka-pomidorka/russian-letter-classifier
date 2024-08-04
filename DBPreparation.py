import os
import cv2
import json
import random
import numpy as np

def pixelization(img):
    height, width = img.shape
    pixels = []

    for y in range(height):
        for x in range(width):
            pixels.append(img[y, x] / 255)
    return pixels

def augmentation(images):
    augmented_images = []

    for image in images:
        # Поворот изображения на случайный угол
        angle = np.random.randint(-15, 15)
        rotated_image = cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1.0), (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))
        
        # Случайное смещение изображения по горизонтали и вертикали
        dx, dy = np.random.randint(-5, 5, 2)
        translated_image = cv2.warpAffine(rotated_image, np.float32([[1, 0, dx], [0, 1, dy]]), (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))
            
        # Изменение яркости и контрастности
        alpha = np.random.uniform(1.0, 1.5)
        beta = np.random.randint(-20, 20)
        brightness_contrast_image = cv2.convertScaleAbs(translated_image, alpha=alpha, beta=beta)
        augmented_images.append(brightness_contrast_image)
            
        # Добавление шума к изображению
        #noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        #noisy_image = cv2.add(brightness_contrast_image, noise)
        #augmented_images.append(noisy_image)

        combined_images = cv2.hconcat([image, brightness_contrast_image])
        combined_images = cv2.resize(combined_images, (1000, 500))

        #cv2.imshow('Origin letter | Updated letter', combined_images)
        #print("Угол: {} \tdx {}\tdy {}\t яркость {}\tконтрастность {}".format(angle, dx, dy, alpha, beta))
        #cv2.waitKey(333)
        #cv2.destroyAllWindows()
    
    return augmented_images

letter_num = []
letter_pic = []
letter_num_for_test = []
letter_pic_for_test = []

for number_of_directory in range(len(os.listdir("DB"))):
    image_directories = []
    images = []
    #aug = []

    directory = "DB/{}".format(number_of_directory)
    os.system('cls||clear')
    print("Дириктория {} из {}".format(directory, len(os.listdir("DB"))-1))

    # Возвращает название файлов с изображениями из всего каталога
    image_directories = os.listdir(directory) 

    random.shuffle(image_directories)
    # записываем сами изображения в список изображений
    images = [cv2.imread("allgr/{}".format(image_directories[pic_path]), cv2.IMREAD_GRAYSCALE) for pic_path in range(len(image_directories))]
    image_directories.clear()
    
    for image in range(len(images)):
        if image >= 500:
            letter_num.append(number_of_directory)
            letter_pic.append(pixelization(images[image]))
        else:
            letter_num_for_test.append(number_of_directory)
            letter_pic_for_test.append(pixelization(images[image]))

    images.clear()


test_data = list(zip(letter_pic_for_test, letter_num_for_test))
random.shuffle(test_data)

letter_pic_for_test.clear()
letter_num_for_test.clear()
for i in range(len(test_data)):
    letter_pic_for_test.append(test_data[i][0])
    letter_num_for_test.append(test_data[i][1])


training_data = list(zip(letter_pic, letter_num))
random.shuffle(training_data)

letter_pic.clear()
letter_num.clear()
for i in range(len(training_data)):
    letter_pic.append(training_data[i][0])
    letter_num.append(training_data[i][1])

path_train = "Training_data"
path_test = "Test_data"
data_Train = {"Pic" : letter_pic,
              "letter_num" : letter_num}
data_Test = {"Pic" : letter_pic_for_test,
             "letter_num" : letter_num_for_test}
file_1 = open(path_test, "w")
file_2 = open(path_train, "w")
print("Сохранение данных для обучения")
json.dump(data_Train, file_1)
print("Сохранение данных для тестирования")
json.dump(data_Test, file_2)
file_1.close()
file_2.close()
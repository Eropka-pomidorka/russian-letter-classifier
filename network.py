import random
import json
import time
import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])] # улучшенная инициализация весов

    # def feedforward(self, a):
    #     for b, w in zip(self.biases, self.weights):
    #         #a = sigmoid(np.dot(w, a) + b)
    #         a = relu(np.dot(w, a) + b)
    #     return a

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights[:-1]):
            a = relu(np.dot(w, a) + b)
        z = np.dot(self.weights[-1], a) + self.biases[-1]
        a = softmax(z)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda = 0.0, test_data=None, callback=None):

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        
        s_time = time.time()
        print(time.strftime('%H:%M:%S', time.localtime()))
        epoch_num = []
        accuracy = []
        for j in range(epochs):
            if callback:
                callback(j + 1)
                
            s_epoch_t = time.time()
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            if test_data:
                f_epoch_t = time.time()
                print("Эпоха {}:\tверных ответов {} из {} | время эпохи {} секунд".format(j,self.evaluate(test_data),n_test, int(f_epoch_t - s_epoch_t)))
                epoch_num.append(j)
                accuracy.append(self.evaluate(test_data)/n_test * 100)
            else:
                print("Эпоха {} завершена".format(j))
        f_time = time.time()
        exe_time(s_time, f_time)
        print(time.strftime('%H:%M:%S', time.localtime()))
        plt.plot(epoch_num, accuracy)
        plt.show()


    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]    
        #self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)] # градиентный спуск
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases =  [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # прямое распространение
        activation = x
        activations = [x]
        list_z = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            list_z.append(z)
            #activation = sigmoid(z)
            activation = relu(z)
            activations.append(activation)

        # выходная ошибка
        #delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(list_z[-1])
        delta = self.cost_derivative(activations[-1], y)
        #delta = (activations[-1] - y) # для случая перекрестной энтропии
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # обратное распространение ошибки
        for l in range(2, self.num_layers):
            z = list_z[-l]
            #sp = sigmoid_prime(z)
            sp = relu_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta # dC/db
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) # dC/dw
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    '''
    # def inference(self):
    #     file = Image.open("custom.png").convert('L')
    #     width, height = file.size
    #
    #     image = []
    #     for y in range(height):    
    #         for x in range(width):
    #             image.append(file.getpixel((x, y)) / 255)
    #     image = np.reshape(image, (-1, 1))
    #
    #     matrix_of_probability = self.feedforward(image)
    #     out = np.argmax(matrix_of_probability)
    #     print("Буква = {} активация нейрона = {}".format(letter(out), matrix_of_probability[out]))
    #     print(matrix_of_probability)
    #
    #     #plt.imshow(image.reshape(28, 28), "grey")
    #     #plt.title(f"Это буква: {letter(out)}")
    #     #plt.show()
    '''
    def inference(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)

        total_images = len(data['Pic'])
        correct_predictions = 0

        s_time = time.time()
        for i in range(total_images):
            image = data['Pic'][i]
            label = data['Num'][i]

            input_data = np.array(image).reshape(-1, 1)

            output = self.feedforward(input_data)

            predicted_class = np.argmax(output)

            if predicted_class == label:
                correct_predictions += 1

        f_time = time.time()

        exe_time(s_time, f_time)
        accuracy = correct_predictions / total_images
        print(f"Inference Accuracy: {accuracy * 100:.2f}%")

    def save(self):
        path = "models\Model"
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        file = open(path, "w")
        json.dump(data, file)
        file.close()

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0, keepdims=True)

def load(filename):
    file = open("{}".format(filename), "r")
    data = json.load(file)
    file.close()
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def letter(token):
    if token == 0:
        letter = "a"
    elif token == 1:
        letter = "б"
    elif token == 2:
        letter = "в"
    elif token == 3:
        letter = "г"
    elif token == 4:
        letter = "д"
    elif token == 5:
        letter = "е"
    elif token == 6:
        letter = "ё"
    elif token == 7:
        letter = "ж"
    elif token == 8:
        letter = "з"
    elif token == 9:
        letter = "и"
    elif token == 10:
        letter = "й"
    elif token == 11:
        letter = "к"
    elif token == 12:
        letter = "л"
    elif token == 13:
        letter = "м"
    elif token == 14:
        letter = "н"
    elif token == 15:
        letter = "о"
    elif token == 16:
        letter = "п"
    elif token == 17:
        letter = "р"
    elif token == 18:
        letter = "с"
    elif token == 19:
        letter = "т"
    elif token == 20:
        letter = "у"
    elif token == 21:
        letter = "ф"
    elif token == 22:
        letter = "х"
    elif token == 23:
        letter = "ц"
    elif token == 24:
        letter = "ч"
    elif token == 25:
        letter = "ш"
    elif token == 26:
        letter = "щ"
    elif token == 27:
        letter = "ъ"
    elif token == 28:
        letter = "ы"
    elif token == 29:
        letter = "ь"
    elif token == 30:
        letter = "э"
    elif token == 31:
        letter = "ю"
    elif token == 32:
        letter = "я"
    else:
        letter = "Ошибка: буквы с номером {} не существует!".format(letter)
    return letter

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return 1. * (z > 0)

def exe_time(start_time, finish_time):
    execution_time = finish_time - start_time
    h, remainder = divmod(execution_time, 3600)
    m, s = divmod(remainder, 60)
    print(f"Время выполнения: {h}ч {m}м {s}с")
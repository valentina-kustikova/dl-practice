import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from torch.utils.data import DataLoader

#Загрузка данных. Необходимо обеспечить демонстрацию избранных изображений и меток классов для подтверждения корректности загрузки.

def main():      
    print("1)Загрузка данных")
    # Функция для демонстрации примеров изображений
    def show_images(images, labels):
        classes = ["самолет", "машина", "птица", "кот", "олень","собака", "лягушка", "лошадь", "корабль", "грузовик"]
        plt.figure(figsize=(12, 6))
        for i in range(10):
            plt.subplot(2, 5, i+1)      #2 строки,5 столбцов
            img = images[i] / 2 + 0.5   
            img = img.permute(1, 2, 0)
            plt.imshow(img)
            plt.title(classes[labels[i]])
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    print("Загружаем данные...")
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    print(f"Данные загружены: {len(train_dataset)} тренировочных, {len(test_dataset)} тестовых")
    
    #показываем примеры
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    show_images(images, labels)
    
#Построение архитектуры сверточной сети. Требуется вывести информацию об архитектуре, возможно выполнить визуализацию графа сети.

    print("2)Архитектура свёрточной сети")
    
    class Convolutional_Neural_Network(nn.Module):
        def __init__(self):
            super(Convolutional_Neural_Network, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding = 1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
            self.fc1 = nn.Linear(64 * 8 * 8, 512)
            self.fc2 = nn.Linear(512, 10)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            #first block
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.pool(x)
            #second block
            x = self.conv2(x)
            x = torch.relu(x)
            x = self.pool(x)

            x = x.view(-1, 64 * 8 * 8)
            x = self.dropout(torch.relu(self.fc1(x)))
            x = self.fc2(x)
            return x
    
    model = Convolutional_Neural_Network()
    print("Сеть создана.")
    
    print("Информация об архитектуре: \n")
    print("Слои сети:")
    print("1. Conv2d(3, 32, 3, padding=1)")
    print("2. ReLU()")
    print("3. MaxPool2d(2, 2)")
    print("4. Conv2d(32, 64, 3, padding=1)")
    print("5. ReLU()")
    print("6. MaxPool2d(2, 2)")
    print("7. Linear(4096, 512)")
    print("8. Dropout(0.5)")
    print("9. Linear(512, 10)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Параметры сети:")
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    
#Обучение модели. Необходимо вывести информацию о параметрах алгоритма обучения. 
#Также по завершении каждой эпохи следует обеспечить вывод точности классификации на тренировочной выборке,
#а по завершении обучения - вывод общего времени обучения.
    print("3)Обучение модели")
    #вычисляет точность модели
    def compute_accuracy(loader, model):
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
    #настройка обучения
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Параметры обучения: \n")
    print(f"Функция потерь: CrossEntropyLoss")
    print(f"Оптимизатор: Adam")
    print(f"Learning rate: 0.001")
    print(f"Количество эпох: 3")
    print(f"Размер батча: 128")
    print("Начинаем обучение...")
    start_time = time.time()
    
    for epoch in range(3):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = compute_accuracy(train_loader, model)
        print(f'Эпоха [{epoch+1}/3]:')
        print(f'Потеря: {epoch_loss:.4f}')
        print(f'Точность: {accuracy:.2f}%')
    
    training_time = time.time() - start_time
    print(f"Обучение завершено за {training_time:.2f} сек")
    
#Тестирование модели. Необходимо обеспечить вывод точности классификации на тестовой выборке,
#по завершении тестирования - вывод среднего времени классификации одного изображения.
    print("4)Тестирование")
    test_start_time = time.time()
    test_accuracy = compute_accuracy(test_loader, model)
    test_time = time.time() - test_start_time
    avg_time_per_image = (test_time / len(test_dataset)) * 1000
    print(f"Точность на тестах: {test_accuracy:.2f}%")
    print(f"Время тестирования: {test_time:.2f} сек")
    print(f"Среднее время классификации: {avg_time_per_image:.4f}")
    model.eval()
    test_iter = iter(test_loader)
    test_images, test_labels = next(test_iter)
    with torch.no_grad():
        outputs = model(test_images)
        _, predicted = torch.max(outputs, 1)
    classes = ["самолет", "машина", "птица", "кот", "олень","собака", "лягушка", "лошадь", "корабль", "грузовик"]
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        img = test_images[i] / 2 + 0.5
        img = img.permute(1, 2, 0)
        plt.imshow(img)
        
        true_label = classes[test_labels[i]]
        pred_label = classes[predicted[i]]
        color = 'green' if test_labels[i] == predicted[i] else 'red'
        
        plt.title(f'Реально: {true_label}\n Предсказано: {pred_label}', color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print("Работа завершена.")
    print(f"Финальная точность: {test_accuracy:.2f}%")

if __name__ == "__main__":
    os.makedirs('./data', exist_ok=True)
    main()
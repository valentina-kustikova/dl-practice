import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
import argparse
import os

def train_model(model, trainloader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        acc_metric = Accuracy(task="multiclass", num_classes=10).to(device)

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            acc_metric.update(outputs, labels)

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = acc_metric.compute().item()
        print(f"Эпоха [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

    return model

def evaluate(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    acc_metric = Accuracy(task="multiclass", num_classes=10).to(device)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            acc_metric.update(outputs, labels)

    avg_test_loss = test_loss / len(dataloader)
    test_acc = acc_metric.compute().item()
    return avg_test_loss, test_acc

def create_transfer_model(model_name, num_classes=10, pretrained=True, freeze_backbone=True):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        backbone_params = model.features.parameters()

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]

    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        backbone_params = model.features.parameters()

    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    if freeze_backbone:
        for param in backbone_params:
            param.requires_grad = False
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        init_info = "ImageNet веса, backbone заморожен"
    else:
        trainable_params = model.parameters()
        init_info = "ImageNet веса, полное обучение"

    return model, trainable_params, init_info

def run_experiment(model_name, freeze_backbone, trainloader, testloader, device, num_epochs, lr, momentum, models_dir):
    model, trainable_params, init_info = create_transfer_model(
        model_name, num_classes=10, pretrained=True, freeze_backbone=freeze_backbone
    )
    model = model.to(device)
    if model_name == 'vgg16':
        modified_layer_info = f"VGG16 Classifier[6]: {model.classifier[6].in_features} -> {model.classifier[6].out_features}"
    elif model_name in ['resnet18', 'resnet50']:
        modified_layer_info = f"ResNet FC: {model.fc.in_features} -> {model.fc.out_features}"
    elif model_name == 'efficientnet_b0':
        modified_layer_info = f"EfficientNet Classifier[1]: {model.classifier[1].in_features} -> {model.classifier[1].out_features}"
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(trainable_params, lr=lr, momentum=momentum)

    model = train_model(model, trainloader, criterion, optimizer, device, num_epochs)

    final_test_loss, final_test_acc = evaluate(model, testloader, criterion, device)
    exp_type = "classifier_only" if freeze_backbone else "full_finetune"
    os.makedirs(models_dir, exist_ok=True)
    suffix = "clf" if freeze_backbone else "full"
    save_path = os.path.join(models_dir, f"{model_name}_{suffix}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Модель сохранена: {save_path}")
    return {
        'model': model_name,
        'experiment': exp_type,
        'accuracy':  final_test_acc,
        'init_info': init_info,
        'optimizer': 'SGD (lr=0.01, momentum=0.9)',
        'modified_layer': modified_layer_info,
        'epochs': num_epochs,
        'save_path': save_path
    }

def plot_comparison_histogram(results, save_path='transfer_learning_comparison.png'):
    labels = []
    accuracies = []
    for r in results:
        exp_label = "clf" if r['experiment'] == 'classifier_only' else "full"
        labels.append(f"{r['model']}\n({exp_label})")
        accuracies.append(r['accuracy'])

    plt.figure(figsize=(14, 6))
    bars = plt.bar(labels, accuracies, color='steelblue')
    plt.ylabel('Точность (Accuracy)', fontsize=12)
    plt.title('Сравнение моделей и стратегий переноса обучения на CIFAR-10', fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def print_detailed_results(results):
    print("Итоговые результаты")

    models_dict = {}
    for result in results:
        model_name = result['model']
        if model_name not in models_dict:
            models_dict[model_name] = []
        models_dict[model_name].append(result)
    
    for model_name, experiments in models_dict.items():
        print(f"\nМОДЕЛЬ: {model_name.upper()}")
        
        modified_layer = experiments[0]['modified_layer']
        print(f"   Модифицированный слой: {modified_layer}")
        print(f"   Эпох обучения: {experiments[0]['epochs']}")
        print(f"   Оптимизатор: {experiments[0]['optimizer']}")
        print()
        
        for exp in experiments:
            exp_type = "Только классификатор" if exp['experiment'] == 'classifier_only' else "Полное дообучение"
            accuracy = exp['accuracy']
            init_info = exp['init_info']
            
            print(f"   - {exp_type}:")
            print(f"      Инициализация: {init_info}")
            print(f"      Точность: {100 * accuracy:.2f}%")

def print_best_result(results):
    if not results:
        return
        
    best_result = max(results, key=lambda x: x['accuracy'])
    
    print("Наилучшая модель")
    print(f"Исходная архитектура: {best_result['model']}")
    print(f"Модифицированный слой: {best_result['modified_layer']}")
    exp_desc = "Обучение только классификатора" if best_result['experiment'] == 'classifier_only' else "Полное дообучение"
    print(f"Тип эксперимента:     {exp_desc}")
    print(f"Инициализация весов:  {best_result['init_info']}")
    print(f"Оптимизатор:          {best_result['optimizer']}")
    print(f"Эпох обучения:        {best_result['epochs']}")
    print(f"Точность на тесте:    {100 * best_result['accuracy']:.2f}%")

def load_model_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        model_names = [line.strip() for line in f if line.strip()]
    return model_names

def cli_argument_parser():
    parser = argparse.ArgumentParser(
        description="Transfer Learning на CIFAR-10 с предобученными моделями."
    )

    parser.add_argument('-dd', '--data_dir',
                        help='Директория для загрузки/хранения датасета CIFAR-10',
                        type=str,
                        dest='data_dir',
                        default='./data')

    parser.add_argument('-e', '--epochs',
                        help='Количество эпох обучения',
                        type=int,
                        dest='epochs',
                        default=10)

    parser.add_argument('-bs', '--batch_size',
                        help='Размер батча для DataLoader',
                        type=int,
                        dest='batch_size',
                        default=64)
    parser.add_argument('-lr', '--learning_rate',
                        help='Скорость обучения для оптимизатора SGD',
                        type=float,
                        dest='learning_rate',
                        default=0.01)

    parser.add_argument('-mom', '--momentum',
                        help='Моментум для оптимизатора SGD',
                        type=float,
                        dest='momentum',
                        default=0.9)

    parser.add_argument('-msp', '--model_save_path',
                        help='Путь для сохранения весов модели',
                        type=str,
                        dest='model_save_path',
                        default='cifar10_cnn.pth')

    parser.add_argument('-psp', '--predictions_save_path',
                        help='Путь для сохранения изображения с предсказаниями',
                        type=str,
                        dest='predictions_save_path',
                        default='transfer_learning_comparison.png')
    
    parser.add_argument('--model_names_file',
                        help='Путь к файлу со списком доступных моделей (по одной на строку)',
                        type=str,
                        dest='model_names_file',
                        default='transfer_models.txt')
    
    parser.add_argument('--models_dir',
                        help='Директория для сохранения весов обученных моделей',
                        type=str,
                        dest='models_dir',
                        default='./saved_models')

    args = parser.parse_args()
    return args

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # transform = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transform)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Обучающих изображений: {len(trainset)}")
    print(f"Тестовых изображений: {len(testset)}")

    model_names = load_model_names(args.model_names_file)
    print(f"Модели для экспериментов: {model_names}")
    results = []

    for model_name in model_names:
        print(f"\nЗапуск экспериментов для {model_name}...")
        try:
            res1 = run_experiment(
                model_name, freeze_backbone=True,
                trainloader=trainloader, testloader=testloader,
                device=device, num_epochs=args.epochs,
                lr=args.learning_rate, momentum=args.momentum,
                models_dir=args.models_dir
            )
            results.append(res1)
            print(f"classifier_only: {res1['accuracy']:.4f}")

            res2 = run_experiment(
                model_name, freeze_backbone=False,
                trainloader=trainloader, testloader=testloader,
                device=device, num_epochs=args.epochs,
                lr=args.learning_rate, momentum=args.momentum,
                models_dir=args.models_dir
            )
            results.append(res2)
            print(f"full_finetune:   {res2['accuracy']:.4f}")
        except Exception as e:
            print(f"Ошибка при обучении {model_name}: {e}")

    if not results:
        print("Ни одна модель не была успешно обучена.")
    else:
        best_result = max(results, key=lambda x: x['accuracy'])

    plot_comparison_histogram(results, save_path=args.predictions_save_path)
    print_detailed_results(results)
    print_best_result(results)

if __name__ == '__main__':
    args = cli_argument_parser()
    main(args)

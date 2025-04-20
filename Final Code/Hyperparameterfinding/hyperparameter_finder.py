import os
import numpy as np
import csv
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import cnn_lstm_model
import cnn_lstm_model_pretrained
import cnn_lstm_model_batchnorm
import video_dataset as dataset


def save_metrics_to_csv(metrics_filepath, config, epoch, phase, loss, accuracy, precision, recall, f1):
    """
    Saves training and validation metrics to a csv file.

    :param string metrics_filepath: The directory to save the metrics to.
    :param dictionary config: Contains the hyperparameter values.
    :param int epoch: Contains the current epoch.
    :param string phase: Contains the phase (e.g. validation).
    :param float loss: The computed loss.
    :param float accuracy: Accuracy score.
    :param float precision: Precision score.
    :param float recall: Recall score.
    :param float f1: F1 score.
    """
    file_exists = os.path.isfile(metrics_filepath)

    with open(metrics_filepath, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Hyperparameters"])
            writer.writerow([f"lr={config['lr_lstm']}", f"batch_size={config['batch_size']}",
                             f"lstm_hidden_size={config['lstm_hidden_size']}",
                             f"cnn_first_output_size={config['cnn_first_output_size']}",
                             f"cnn_final_output_size={config['cnn_final_output_size']}",
                             f"cnn_dropout_rate={config['cnn_dropout_rate']}",
                             f"lstm_dropout_rate={config['lstm_dropout_rate']}"])
            writer.writerow([])
            writer.writerow(['Epoch', 'Phase', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

        writer.writerow([epoch, phase, loss, accuracy, precision, recall, f1])

def train_and_validate_model(config, checkpoint_dir=None):
    """
    Trains and validates a given CNN-LSTM model using the specified configuration parameters.

    :param dictionary config: Dictionary containing the hyperparameters.
    :param checkpoint_dir: (for integration with Ray Tune).
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((225, 400)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    root_dir = "DIRECTORY TO CLIPS"
    whole_dataset = dataset.Dataset_BA(root_dir, transform=transform)

    all_labels = [sample[1] for sample in whole_dataset.samples]

    #Split into Val (15%) and Train (85%)
    test_split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=25)
    train_indices, val_indices = next(test_split.split(np.zeros(len(all_labels)), all_labels))

    val_dataset = Subset(whole_dataset, val_indices)
    print(len(val_dataset))
    train_dataset = Subset(whole_dataset, train_indices)
    print(len(train_dataset))

    model_path = "DIRECTORY TO MODEL"
    os.makedirs(model_path, exist_ok=True)

    csv_filename = f"metrics_lr{config['lr_pretrained']}_bs{config['batch_size']}_lstm{config['lstm_hidden_size']}_cnn{config['cnn_first_output_size']}-{config['cnn_final_output_size']}.csv"
    metrics_filepath = os.path.join(model_path, csv_filename)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = int(config["batch_size"]), shuffle=True, num_workers=6)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = int(config["batch_size"]), shuffle=False, num_workers=6)

    model = cnn_lstm_model_pretrained.CNN_LSTM(
        lstm_hidden_size=int(config["lstm_hidden_size"]),
        cnn_final_output_size=int(config["cnn_final_output_size"]),
        lstm_dropout_rate=config["lstm_dropout_rate"]
    )

    model.to(device)

    loss_function = nn.BCELoss()
    #For base/batchnorm
    #optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    #for Pretrained
    cnn_params = list(model.cnn.parameters())
    lstm_params = list(model.lstm.parameters()) + list(model.fc.parameters())
    optimizer = torch.optim.Adam([
        {'params': cnn_params, 'lr': config["lr_pretrained"]},  # Low learning rate for pretrained CNN
        {'params': lstm_params, 'lr': config["lr_lstm"]},  # Higher learning rate for LSTM and FC
    ])

    epochs = 10
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch, (inputs, labels) in enumerate(train_loader):

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if (batch+1) % 100 == 0:
                print(f"Current Batch: {batch}, current loss: {loss.item()}")

        average_train_loss = train_loss / len(train_loader)

        model.eval()
        validation_loss = 0
        all_predictions, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs).squeeze()

                loss = loss_function(outputs, labels)

                validation_loss += loss.item()

                predictions = (outputs >= 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())


        average_validation_loss = validation_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_precision = precision_score(all_labels, all_predictions, zero_division=1)
        val_recall = recall_score(all_labels, all_predictions, zero_division=1)
        val_f1 = f1_score(all_labels, all_predictions, zero_division=1)

        save_metrics_to_csv(metrics_filepath, config, epoch, 'validation', average_validation_loss, val_accuracy, val_precision, val_recall, val_f1)

        tune.report({"loss": average_validation_loss, "accuracy": val_accuracy, "train_loss":average_train_loss})


if __name__ == "__main__":
    #Inspiration: https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html#tune-pytorch-cifar-ref
    analysis = tune.run(
        train_and_validate_model,
        resources_per_trial={"cpu": 32, "gpu": 1},
        config={
            #"lr":tune.choice([1e-5, 5e-5, 1e-4, 5e-4]),    #For base/batchnorm
            "lr_pretrained" : tune.choice([1e-5, 5e-5]),    #for pretrained
            "lr_lstm": tune.choice([1e-5, 5e-5, 1e-4, 5e-4]),    #for pretrained
            "batch_size": tune.choice([16]),
            "lstm_hidden_size": tune.choice([64, 128]),
            "cnn_first_output_size": tune.choice([16, 32, 64]),
            "cnn_final_output_size": tune.choice([64, 128, 256]),
            "cnn_dropout_rate": tune.uniform(0.3, 0.6),
            "lstm_dropout_rate": tune.uniform(0.1, 0.5)
        },
        num_samples=4,
        scheduler=ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=10,           #Max num Epochs
            grace_period=1,
            reduction_factor=2
        ),
        storage_path="DIRECTORY FOR RAY TUNE DATA",
        name="NAME_TODO_CHANGE"
    )

    print("Best hyperparameters found were: ", analysis.best_config)
    try:
        print(analysis.results_df())
    except Exception as e:
        print(f"Error retrieving results: {e}")
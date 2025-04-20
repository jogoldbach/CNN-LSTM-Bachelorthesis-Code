import csv
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from torchvision import transforms

import cnn_lstm_model
import video_dataset as dataset
import video_dataset_no_augmentations as datasetNoAugs

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, loss_function, optimizer, model_save_path, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model_save_path = model_save_path
        self.device = device
        self.train_losses = []
        self.validation_losses = []
        self.best_validation_loss = float('inf')

    def save_metrics_to_csv(self, metrics_filepath, epoch, batch, phase, loss, accuracy, precision, recall, f1):
        """
        Saves training and validation metrics to a csv file.

        :param string metrics_filepath: The directory to save the metrics to.
        :param int epoch: Contains the current epoch.
        :param int batch: Contains the current batch.
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
                writer.writerow(['Epoch', 'Batch', 'Phase', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

            writer.writerow([epoch, batch, phase, loss, accuracy, precision, recall, f1])


    def train_one_epoch(self, epoch_number, metrics_path_epoch, metrics_path_batch):
        """
        Trains the model for a single epoch. This is important for Early Stopping

        :param int epoch_number: The current epoch index.
        :param string metrics_path_epoch: Path to save epoch-level metrics.
        :param string metrics_path_batch: Path to save batch-level metrics.
        :return: The loss and accuracy is returned.
        """

        self.model.train()    #Training Mode
        epoch_loss = 0
        all_predictions, all_lables = [], []
        total_steps = len(self.train_loader)
        for batch, (inputs, labels) in enumerate(self.train_loader):

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            #Clears old gradients
            self.optimizer.zero_grad()

            #Modell pass
            outputs = self.model(inputs).squeeze()

            #Compute gradients
            loss = self.loss_function(outputs, labels)
            loss.backward()

            #Update weights
            self.optimizer.step()

            labels_from_batch = labels.cpu().numpy()
            predictions = (outputs >= 0.5).float()
            predictions_from_batch = predictions.cpu().numpy()

            train_loss_batch = loss.item()
            train_accuracy_batch = accuracy_score(labels_from_batch, predictions_from_batch)

            self.save_metrics_to_csv(metrics_path_batch, (epoch_number + 1), (batch + 1), 'train', train_loss_batch,
                                     train_accuracy_batch, 0, 0, 0)

            epoch_loss += train_loss_batch

            all_predictions.extend(predictions_from_batch)
            all_lables.extend(labels_from_batch)

            #Prints Metrics for every tenth batch
            if (batch + 1) % 10 == 0:
                print(f"Epoch:{epoch_number + 1};     Batch: {batch + 1} / {total_steps};    Loss: {train_loss_batch:>4f};     Accuracy: {train_accuracy_batch:>4f}")


        train_loss_epoch = epoch_loss / len(self.train_loader)
        train_accuracy_epoch = accuracy_score(all_lables, all_predictions)
        train_precision = precision_score(all_lables, all_predictions, zero_division=1)
        train_recall = recall_score(all_lables, all_predictions, zero_division=1)
        train_f1 = f1_score(all_lables, all_predictions, zero_division=1)

        self.save_metrics_to_csv(metrics_path_epoch, (epoch_number + 1), 0, 'train', train_loss_epoch,
                                 train_accuracy_epoch, train_precision, train_recall, train_f1)

        return train_loss_epoch, train_accuracy_epoch


    def validate(self, metrics_path, epoch):
        """
        Evaluates the model on the validation set.

        :param string metrics_path:
        :param int epoch: The current epoch index.
        :return: The loss and accuracy is returned.
        """

        self.model.eval()    #Evaluation Mode
        validation_loss = 0
        all_predictions, all_lables = [], []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                #Modell pass
                outputs = self.model(inputs).squeeze()

                loss = self.loss_function(outputs, labels)
                validation_loss += loss.item()

                predictions = (outputs >= 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_lables.extend(labels.cpu().numpy())

        validation_loss = validation_loss / len(self.val_loader)
        validation_accuracy = accuracy_score(all_lables, all_predictions)
        validation_precision = precision_score(all_lables, all_predictions, zero_division=1)
        validation_recall = recall_score(all_lables, all_predictions, zero_division=1)
        validation_f1 = f1_score(all_lables, all_predictions, zero_division=1)

        self.save_metrics_to_csv(metrics_path, (epoch + 1), 0, 'val', validation_loss, validation_accuracy,
                                 validation_precision, validation_recall, validation_f1)

        return validation_loss, validation_accuracy

    def save_model(self, model_name):
        """
        Saves a given model to a file.

        :param string model_name: The file name of the model to save.
        """

        model_path = os.path.join(self.model_save_path, model_name)
        torch.save(self.model.state_dict(), model_path)    #https://pytorch.org/tutorials/beginner/saving_loading_models.html


    def train(self, number_of_epochs, early_stopping_patience, metrics_path_epoch, metrics_path_batch, model_name):
        """
        Trains the model over multiple epochs with early stopping.

        :param number_of_epochs: Maximum number of epochs to train.
        :param early_stopping_patience: Number of epochs to wait before early stopping if there are no improvement.
        :param metrics_path_epoch: Path to save epoch-level metrics.
        :param metrics_path_batch: Path to save batch-level metrics.
        :param model_name: File name for saving the best model.
        """

        patience = 0

        for epoch in range(number_of_epochs):
            train_loss, train_accuracy = self.train_one_epoch(epoch, metrics_path_epoch, metrics_path_batch)

            validation_loss, validation_accuracy = self.validate(metrics_path_epoch, epoch)

            self.train_losses.append(train_loss)
            self.validation_losses.append(validation_loss)

            print('Epoch {}/{}'.format(epoch + 1, number_of_epochs))
            print('Training Loss: {:.4f}, Training Accuracy {:.4f}:'.format(train_loss, train_accuracy))
            print('Validation Loss: {:.4f}, Validation Accuracy {:.4f}:'.format(validation_loss, validation_accuracy))

            #Updating the best loss or patience
            if validation_loss < self.best_validation_loss:
                self.best_validation_loss = validation_loss
                patience = 0
                print("New best validation loss: {:.4f}".format(validation_loss))
                self.save_model(model_name)
            else:
                patience += 1


            if patience >= early_stopping_patience:
                print("Early stopping patience reached")
                break
        print("Training Finished!")


    def test(self, metrics_path, model_name):
        """
        Loads the best model and evaluates it on the test dataset.

        :param metrics_path: Path to save test metrics.
        :param model_name: File name for saving the best model.
        :return: The loss and accuracy is returned.
        """

        model_path = os.path.join(self.model_save_path, model_name)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        test_loss = 0
        all_predictions, all_lables, all_scores = [], [], []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs).squeeze()
                loss = self.loss_function(outputs, labels)

                test_loss += loss.item()
                predictions = (outputs >= 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_lables.extend(labels.cpu().numpy())
                all_scores.extend(outputs.cpu().numpy())

        test_loss = test_loss / len(self.test_loader)
        test_accuracy = accuracy_score(all_lables, all_predictions)
        test_precision = precision_score(all_lables, all_predictions, zero_division=1)
        test_recall = recall_score(all_lables, all_predictions, zero_division=1)
        test_f1 = f1_score(all_lables, all_predictions, zero_division=1)

        print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(test_loss, test_accuracy))

        self.save_metrics_to_csv(metrics_path, 0, 0, 'test', test_loss, test_accuracy, test_precision,
                                 test_recall, test_f1)

        return test_loss, test_accuracy

    def plot_learning_curve(self):
        """
        Plots the learning curve of training and validation loss over the epochs.

        """

        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 5))

        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.validation_losses, label='Validation Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Curve')

        plt.legend()
        plt.show()



model = cnn_lstm_model.CNN_LSTM()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

loss_function = nn.BCELoss()    #Binary cross entropy loss

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)    #L2-Regularization


if __name__ == "__main__":    #Bestpractice for Multiprocessing (num_workers > 0)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((225, 400)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    root_dir_train = "DIRECTORY TO TRAINING DATA OF RANDOM SPLIT"
    root_dir_test_val = "DIRECTORY TO TEST AND VALIDATION DATA OF RANDOM SPLIT"

    train_dataset = dataset.Dataset_BA(root_dir_train, transform=transform, device=device)
    print(len(train_dataset))

    test_val_dataset = datasetNoAugs.Dataset_BA(root_dir_test_val, transform=transform)
    print(len(test_val_dataset))

    test_train_lables = [sample[1] for sample in test_val_dataset.samples]

    #StratifiedShuffleSplit into test (50%) and val (50%)
    train_val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=25)
    val_indices, test_indices = next(train_val_split.split(np.zeros(len(test_train_lables)), test_train_lables))

    test_dataset = Subset(test_val_dataset, test_indices)
    print(len(test_dataset))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=6)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)

    val_dataset = Subset(test_val_dataset, val_indices)
    print(len(val_dataset))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=6)

    model_path = "DIRECTORY TO MODEL"
    model_name = "NAME OF MODEL FILE"

    os.makedirs(model_path, exist_ok=True)

    metrics_path_epoch = "DIRECTORY TO CSV FILE FOR EPOCHS"
    metrics_path_batch = "DIRECTORY TO CSV FILE FOR BATCHES"

    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                      loss_function=loss_function, optimizer=optimizer, model_save_path=model_path, device=device)

    trainer.train(200,5, metrics_path_epoch, metrics_path_batch, model_name)

    trainer.plot_learning_curve()

    trainer.test(metrics_path_epoch, model_name)


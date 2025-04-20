import os
import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from torchvision import transforms

import cnn_lstm_model
import cnn_lstm_model_batchnorm
import cnn_lstm_model_pretrained

import video_dataset_adverserial as datasetAdv

class Tester:
    def __init__(self, model, test_loader, loss_function, model_save_path, device):
        self.model = model
        self.test_loader = test_loader
        self.loss_function = loss_function
        self.model_save_path = model_save_path
        self.device = device
        self.train_losses = []
        self.validation_losses = []
        self.best_validation_loss = float('inf')

    def test(self, model_name):
        """
        This method is used to test the trained model on adverserial data.

        :param string model_name: The name of the model file.
        :return: loss and accuracy is returned.
        """
        model_path = os.path.join(self.model_save_path, model_name)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        
        test_loss = 0
        all_predictions, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs).squeeze()
                loss = self.loss_function(outputs, labels)

                test_loss += loss.item()
                predictions = (outputs >= 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())


        test_loss = test_loss / len(self.test_loader)
        test_accuracy = accuracy_score(all_labels, all_predictions)
        test_precision = precision_score(all_labels, all_predictions, zero_division=1)
        test_recall = recall_score(all_labels, all_predictions, zero_division=1)
        test_f1 = f1_score(all_labels, all_predictions, zero_division=1)

        print('Adverserial Test Loss: {:.4f}, Adverserial Test Accuracy: {:.4f}, Adverserial Test Precision: {:.4f}, Adverserial Test Recall: {:.4f}, Adverserial Test F1-Score: {:.4f},'.format(test_loss, test_accuracy, test_precision, test_recall, test_f1))


        return test_loss, test_accuracy

model = cnn_lstm_model_pretrained.CNN_LSTM()

device = torch.device('cuda' if torch.cuda.is_availabel() else 'cpu')

model = model.to(device)

loss_function = nn.BCELoss()


if __name__ == "__main__":    #Bestpractice for Multiprocessing (num_workers > 0)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((225, 400)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

 
    root_dir_test_val = "DIRECTORY TO TEST AND VALIDATION DATA"


    test_val_dataset = datasetAdv.Dataset_BA(root_dir_test_val, transform=transform)
    print(len(test_val_dataset))

    test_train_labels = [sample[1] for sample in test_val_dataset.samples]

    #Split into Test (50%) and Val (50%)
    train_val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=25)
    _, test_indices = next(train_val_split.split(np.zeros(len(test_train_labels)), test_train_labels))
    
    test_dataset = Subset(test_val_dataset, test_indices)
    print(len(test_dataset))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=6)

    model_path = "DIRECTORY TO MODEL"
    model_name = "NAME OF MODEL FILE"

    os.makedirs(model_path, exist_ok=True)

    tester = Tester(model=model, test_loader=test_loader, loss_function=loss_function, model_save_path=model_path, device=device)

    tester.test(model_name)
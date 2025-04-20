import csv
import os
import time
import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchviz import make_dot
from torchvision import transforms

import cnn_lstm_model
import cnn_lstm_model_batchnorm
import cnn_lstm_model_pretrained

import video_dataset as dataset

class Tester:
    def __init__(self, model, test_loader,model_save_path, loss_function,device):
        self.model = model
        self.test_loader = test_loader
        self.model_save_path = model_save_path
        self.device = device
        self.loss_function = loss_function

    def save_metrics_to_csv(self,metrics_filepath, epoch, batch, phase, loss, accuracy, precision, recall, f1):
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

        #Opens file in append mode
        with open(metrics_filepath, mode='a', newline='') as file:
            writer = csv.writer(file)

            #First row
            if not file_exists:
                writer.writerow(['Epoch','Batch', 'Phase', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

            writer.writerow([epoch, batch, phase, loss, accuracy, precision, recall, f1])

    def test(self, model_name):
        """
        This method is used to test the trained model on the new dataset.

        :param string model_name: The name of the model file.
        :return: The loss and accuracy is returned.
        """
        model_path = os.path.join(self.model_save_path, model_name)
        self.model.load_state_dict(torch.load(model_path,weights_only=True))
        self.model.eval()
        test_loss = 0
        all_predictions, all_labels, all_times = [], [], []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                start_time = time.time()
                test = self.model(inputs)
                make_dot(test.mean(), params=dict(model.named_parameters()))
                end_time = time.time()
                all_times.append((end_time - start_time)*1000)

                outputs = test.squeeze(1)
                loss = self.loss_function(outputs, labels)

                test_loss += loss.item()
                predictions = (outputs >= 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss = test_loss / len(self.test_loader)
        test_accuracy = accuracy_score(all_labels, all_predictions)
        test_precision = precision_score(all_labels, all_predictions,zero_division=1)
        test_recall = recall_score(all_labels, all_predictions,zero_division=1)
        test_f1 = f1_score(all_labels, all_predictions,zero_division=1)

        print('Test Loss: {:.4f}, Test Accuracy {:.4f}, Test precision {:.4f}, recall {:.4f} , test_f1 {:.4f}'.format(test_loss, test_accuracy, test_precision, test_recall, test_f1))
        print('Average Time: {:.4f}ms, Max Time: {:.4f}ms, Min Time: {:.4f}ms'.format(np.mean(all_times), np.max(all_times), np.min(all_times)))
        #self.save_metrics_to_csv(metrics_path, 0, 0,'test different robot', test_loss, test_accuracy, test_precision, test_recall, test_f1)

        return test_loss, test_accuracy

model = cnn_lstm_model.CNN_LSTM()

device = torch.device('cuda' if torch.cuda.is_availabel() else 'cpu')

model = model.to(device)

loss_function = nn.BCELoss()

if __name__ == "__main__":    #Bestpractice for Multiprocessing (num_workers > 0)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((400, 225)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    root_dir = "DIRECTORY TO LABELED VIDEO CLIPS"

    model_path = "DIRECTORY TO MODEL"
    model_name = "NAME OF MODEL FILE"


    whole_dataset = dataset.Dataset_BA(root_dir, transform=transform)

    test_loader=torch.utils.data.DataLoader(dataset=whole_dataset, batch_size=1, shuffle=False, num_workers=6)

    tester = Tester(model,test_loader,model_path,loss_function, device)
    tester.test(model_name)
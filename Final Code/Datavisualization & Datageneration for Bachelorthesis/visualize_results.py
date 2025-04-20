import matplotlib.pyplot as plt
import pandas as pd

class VisualizeResults:
    def __init__(self):
        pass

    def read_CSV_file(self, csvFilePath):
        """
        Reads a CSV file into a pandas DataFrame.

        :param string csvFilePath: Path to the CSV file.
        :return: Pandas DataFrame of the CSV file.
        """

        csvFile = pd.read_csv(csvFilePath)
        return csvFile


    def plot_loss_and_accur_metrics_epochs(self, dataframe, model_type):
        """
        Plots the training and validation loss and accuracy over epochs from the given DataFrame.

        :param Pandas DataFrame dataframe: DataFrame containing data metrics.
        :param string model_type: Name of the model.
        """

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        axes = axes.flatten()
        metrics = ['Loss', 'Accuracy']

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            for phase in ['train', 'val']:
                phase_data = dataframe[dataframe['Phase'] == phase]
                ax.plot(phase_data['Epoch'], phase_data[metric], label=f'{phase}')
            ax.set_title(f'{metric} over Epochs')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True)


        plt.tight_layout()

        plt.savefig(f'visualized_results_acc_and_loss_{model_type}.png', dpi=300, bbox_inches='tight')

        plt.show()



v = VisualizeResults()

csv_path= 'DIRECTORY TO CSV FILE'

csv=v.read_CSV_file(csv_path)

v.visualize_results_acc_and_loss_epochs(csv_path,'NAME OF THE TYPE OF MODEL')





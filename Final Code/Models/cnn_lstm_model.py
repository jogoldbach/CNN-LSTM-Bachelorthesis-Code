import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, lstm_hidden_size=128, cnn_first_output_size=16, cnn_final_output_size=64, num_classes=1,
                 cnn_dropout_rate=0.54, lstm_dropout_rate=0.23):
        """
        Initializes the CNN-LSTM model for processing sequences of image frames (Videos).

        The CNN extracts spatial features from each frame, while the LSTM captures
        temporal features across the sequence of frames.

        https://discuss.pytorch.org/t/cnn-lstm-for-video-classification/185303/7

        :param int, optional lstm_hidden_size: Hidden state size of the LSTM (default is 128).
        :param int, optional cnn_first_output_size: Number of output Channels after the first convolutional layer (default is 16).
        :param int, optional cnn_final_output_size: Final output size from the CNN, and input size for the LSTM (default is 64).
        :param int, optional num_classes: Number of output Classes (default is 1).
        :param float, optional cnn_dropout_rate: Dropout probability applied to CNN layers (default is 0.54).
        :param float, optional lstm_dropout_rate: Dropout probability applied to LSTM layers (default is 0.23).
        """

        super(CNN_LSTM, self).__init__()
        self.NAME = 'CNN LSTM Normal'

        #CNN to extract spacial features sequentially https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=cnn_first_output_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=cnn_dropout_rate),

            nn.Conv2d(in_channels=cnn_first_output_size, out_channels=cnn_first_output_size*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=cnn_dropout_rate),

            nn.Flatten(),
            nn.Linear(in_features=cnn_first_output_size*2*56*56, out_features=cnn_final_output_size),
            nn.ReLU(),
            nn.Dropout(p=cnn_dropout_rate),
        )

        #LSTM to extract time relevant features
        self.lstm = nn.LSTM(input_size=cnn_final_output_size, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        self.lstm_dropout = nn.Dropout(p=lstm_dropout_rate)

        self.fc = nn.Linear(in_features=lstm_hidden_size, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass of the CNN-LSTM model.

        :param torch.Tensor x: Input tensor of shape (batch_size, num_frames, channels, height, width), representing the input frames.
        :return torch.Tensor output: Output predictions.
        """

        batch_size, num_frames, C, H, W = x.size()

        #batch_size * num_frames : so that CNN can view/process each frame at a time
        x = x.view(batch_size * num_frames, C, H, W)

        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, num_frames, -1)

        lstm_out, _ = self.lstm(cnn_features)
        lstm_out = self.lstm_dropout(lstm_out)

        #Get the last hiddenlayer from LSTM for classification
        #https://medium.com/@hkabhi916/understanding-lstm-for-sequence-classification-a-practical-guide-with-pytorch-ac40e84ad3d5
        lstm_out_last = lstm_out[:,-1,:]
        output=self.fc(lstm_out_last)
        output = self.sigmoid(output)

        return output
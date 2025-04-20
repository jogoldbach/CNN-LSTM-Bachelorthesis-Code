import torch.nn as nn
import torchvision.models as models

class CNN_LSTM(nn.Module):
    def __init__(self, lstm_hidden_size=128, cnn_final_output_size=128, num_classes=1, lstm_dropout_rate=0.37, freeze_cnn=True):
        """
        Initializes the CNN-LSTM model for processing sequences of image frames (Videos).

        The EfficientNet CNN extracts spatial features from each frame, while the LSTM captures
        temporal features across the sequence of frames.

        https://discuss.pytorch.org/t/cnn-lstm-for-video-classification/185303/7

        :param int, optional lstm_hidden_size: Hidden state size of the LSTM (default is 128).
        :param int, optional cnn_final_output_size: Final output size from the CNN, and input size for the LSTM (default is 128).
        :param int, optional num_classes: Number of output Classes (default is 1).
        :param float, optional lstm_dropout_rate: Dropout probability applied to LSTM layers (default is 0.37).
        :param bool, optional freeze_cnn: If True, the pretrained CNN model will be frozen (default is True).
        """
        super(CNN_LSTM, self).__init__()
        self.NAME = 'CNN LSTM PreTrained'

        #Pretrained CNN to extract spacial features sequentially
        self.cnn = models.efficientnet_b0(weights='DEFAULT')    #Fast computing Model (5.3M Params) Paper: https://arxiv.org/pdf/1905.11946

        self.cnn.classifier = nn.Sequential(
            nn.Linear(in_features=self.cnn.classifier[1].in_features, out_features=cnn_final_output_size),
            nn.ReLU()
        )

        # Optionally freeze early CNN layers
        if freeze_cnn:
            for param in self.cnn.features.parameters():
                param.requires_grad = False

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

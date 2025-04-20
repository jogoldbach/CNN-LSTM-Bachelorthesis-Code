from torchviz import make_dot

import cnn_lstm_model_batchnorm
import cnn_lstm_model
import cnn_lstm_model_pretrained


import torch

"""
This script is used to visualize the models.

"""

model = cnn_lstm_model_pretrained.CNN_LSTM()
y = torch.zeros([1, 30, 3, 224, 224])

dot = make_dot(model(y), params=dict(model.named_parameters()))
dot.render("model_visualization_pretrained", format="pdf")


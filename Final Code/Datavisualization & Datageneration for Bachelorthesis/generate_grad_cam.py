import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import cnn_lstm_model
import cnn_lstm_model_batchnorm
import cnn_lstm_model_pretrained
import video_dataset as dataset
from torchvision import transforms
import video_dataset_adverserial
import random


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        #Register hooks to save the gradients.
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    #The activations and gradient functions are passed to PyTorch
    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x):
        """
        Creates the heatmaps for Grad-CAM.

        :param tensor x: Input-Video as Tensor in the form of (B, T, C, H, W) (B = Batch, T = Number of frames, C = Number of channels)
        :return: The heatmaps.
        """

        #Activation for the gradients
        x.requires_grad = True

        output = self.model(x)

        class_idx = torch.argmax(output, dim=1).item()

        #Set gradients to zero and make a backwardpass for the target class
        self.model.zero_grad()
        output[:, class_idx].backward(retain_graph=True)

        activations = self.activations
        gradients = self.gradients

        if len(activations.shape) == 5:
            B, T, C, H_feat, W_feat = activations.shape
            activations = activations.view(B * T, C, H_feat, W_feat)
            gradients = gradients.view(B * T, C, H_feat, W_feat)
        else:
            T = activations.shape[0]

        cams = []

        for i in range(T):
            act = activations[i].detach().cpu().numpy()
            grad = gradients[i].detach().cpu().numpy()


            weights = np.mean(grad, axis=(1, 2))
            cam = np.zeros(act.shape[1:], dtype=np.float32)
            for j, w in enumerate(weights):
                cam += w * act[j]

            cam = np.maximum(cam, 0)

            orig_h = x.shape[-2]
            orig_w = x.shape[-1]
            cam = cv2.resize(cam, (orig_w, orig_h))


            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cams.append(cam)

        return cams



def overlay_heatmap(img, heatmap, alpha=0.4):
    """
    Overlays a Grad-CAM heatmap on top of an image.

    :param RGB img: The original image.
    :param Numpy Array heatmap: The generated Grad-CAM heatmap.
    :param alpha: The alpha blending factor that dictates the transparency of the overlayed heatmap.
    :return: An overlayed image.
    """

    #Convert Heatmap to colormap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed


if __name__ == '__main__':

    model = cnn_lstm_model.CNN_LSTM()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    model_path = "DIRECTORY TO MODEL FOR GRAD-CAM ANALYSIS"

    model.load_state_dict(torch.load(model_path))

    model.eval()



    #Select the target layers depending on the model
    if model.NAME == 'CNN LSTM PreTrained':
        target_layer = model.cnn.features[8][2]
    elif model.NAME == 'CNN LSTM Normal':
        target_layer = model.cnn[4]
    else:
        target_layer = model.cnn[8]

    print(target_layer)

    #Create the GradCAM object.
    grad_cam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((225, 400)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    root_dir = "DIRECTORY TO GRAD-CAM VIDEOS"

    whole_dataset = dataset.Dataset_BA(root_dir, transform=transform)
    print(len(whole_dataset))


    all_labels = [sample[1] for sample in whole_dataset.samples]

    data_loader = torch.utils.data.DataLoader(whole_dataset, batch_size=1, shuffle=False, num_workers=6)

    for idx, (video, label) in enumerate(data_loader):
        video = video.to(device)
        model.train()
        heatmaps = grad_cam(video)    #This will be a list of 30 heatmaps.

        #Convert the video clip tensor to a numpy array for visualization.
        video_np = video.squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy()
        video_np = (video_np - video_np.min()) / (video_np.max() - video_np.min() + 1e-8)
        video_np = (video_np * 255).astype(np.uint8)

        #Overlay the heatmap and store the result frame-wise.
        overlayed_frames = []
        for frame, heatmap in zip(video_np, heatmaps):
            overlay = overlay_heatmap(frame, heatmap)
            overlayed_frames.append(overlay)

        #Display frame as an example.
        plt.imshow(overlayed_frames[0])
        plt.axis('off')
        plt.title('Grad-CAM Overlay on Frame 0')
        plt.show()

        #Save overlayed frames as a video.
        output_video_file = f'grad_cam_overlay_{idx:02d}.mp4'
        fps = 30
        height, width, _ = overlayed_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

        for frame in overlayed_frames:
            #Convert frame from RGB to BGR (OpenCV uses BGR).
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()

        #Save frames as images.
        for idxFrame, frame in enumerate(overlayed_frames):
            cv2.imwrite(f'overlay_frame_{idxFrame:02d}_video_{idx:02d}.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class CreateUnifiedPicture:
    def __init__(self):
        pass


    def plot_images_from_directory(self, foto_dir,  cols, rows, headline="Bilderübersicht"):
        """
        Displays images from a specified directory in a grid layout.

        :param string foto_dir: The directory to display images from.
        :param cols: Number of columns to display in the grid.
        :param rows: Number of rows to display in the grid.
        :param headline: Title displayed above the grid of images (default is "Bilderübersicht").
        """

        valid_extensions = ('.png', '.jpg', '.jpeg',)
        image_files = [f for f in os.listdir(foto_dir) if f.lower().endswith(valid_extensions)]

        fig, axs  = plt.subplots(rows, cols)

        axs  = axs.flatten()

        for idx, img_file in enumerate(image_files):
            ax = axs[idx]
            img_dir = os.path.join(foto_dir, img_file)
            img = mpimg.imread(img_dir)
            ax.imshow(img)
            ax.set_title(img_file.split("_")[5], fontsize=10)
            ax.axis('off')

        #Remove the axis for unused subplot
        for square in axs[len(image_files):]:
            square.axis('off')


        plt.suptitle(headline)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)    #Space for the Headline
        plt.show()


    plot_images_from_directory("CHANGE TO FOTO DIRECTORY", 2,3,"Grad-CAM Heatmaps des selben Bildes eines Aktiv-Videos von allen Modellen.")
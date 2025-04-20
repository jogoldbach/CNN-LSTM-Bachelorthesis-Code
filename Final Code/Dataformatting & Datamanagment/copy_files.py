import os
import shutil
from sklearn.model_selection import train_test_split

#Directory where the videos are
source_dir = 'CHANGE TO SOURCE DIRECTORY'

#Target directories
test_dir = 'CHANGE TO TEST DIRECTORY'
train_dir = 'CHANGE TO TRAIN DIRECTORY'


os.makedirs(os.path.join(test_dir, 'Active'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'Idle'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'Active'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'Idle'), exist_ok=True)

#Get Videos
#https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
active_videos = [f for f in os.listdir(os.path.join(source_dir, 'Active'))]
idle_videos = [f for f in os.listdir(os.path.join(source_dir, 'Idle'))]

total_number_videos = len(active_videos) + len(idle_videos)

#Change for desired size
train_size = 16034
test_size = 6874

#Creating the Stratified Splits
active_train_size = int((train_size / total_number_videos) * len(active_videos))
idle_train_size = train_size - active_train_size

active_train, active_test = train_test_split(active_videos, train_size=active_train_size, random_state=25)
idle_train, idle_test = train_test_split(idle_videos, train_size=idle_train_size, random_state=25)


def copy_files(file_list, source_folder, destination_folder):
    """
    This function copies files from source_folder to destination_folder.

    :param list of strings file_list: List of files to be copied.
    :param string source_folder: Source folder directory.
    :param string destination_folder: Destination folder directory.
    """
    for file in file_list:
        source_dir = os.path.join(source_folder, file)
        target_dir = os.path.join(destination_folder, file)
        shutil.copy(source_dir, target_dir)


copy_files(active_test, os.path.join(source_dir, 'Active'), os.path.join(test_dir, 'Active'))
copy_files(idle_test, os.path.join(source_dir, 'Idle'), os.path.join(test_dir, 'Idle'))

copy_files(active_train, os.path.join(source_dir, 'Active'), os.path.join(train_dir, 'Active'))
copy_files(idle_train, os.path.join(source_dir, 'Idle'), os.path.join(train_dir, 'Idle'))

print("The videos where copied and sorted into training and testsets.")
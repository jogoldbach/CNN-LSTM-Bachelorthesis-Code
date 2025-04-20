import os
import shutil


def sort_clips_by_annotation(video_path, output_dir):
    """
    Used to sort clips by its annotation ('Active', 'Idle') into the relevant folder.

    :param video_path: Directory to the unsorted video clips.
    :param output_dir: Path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    original_video_path = video_path

    for folder_name in os.listdir(video_path):

        video_path = original_video_path

        folder_path = os.path.join(video_path, folder_name)

        video_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]

        for video in video_files:
            video_path = os.path.join(folder_path, video)

            if "Idle" in video:
                target_path = os.path.join(output_dir, "Idle")
                os.makedirs(target_path, exist_ok=True)
            else:
                target_path = os.path.join(output_dir, "Active")
                os.makedirs(target_path, exist_ok=True)

            #Move the video to the target directory
            shutil.move(video_path, target_path)


video_file = "DIRECTORY TO VIDEO CLIPS"
output_folder = "DIRECTORY TO TARGET FOLDER"
sort_clips_by_annotation(video_file, output_folder)
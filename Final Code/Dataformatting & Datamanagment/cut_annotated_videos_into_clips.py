import cv2
import json
import os


def cut_video_by_annotations(video_path, json_path, output_dir):
    """
    Cuts videos into clips by iterating given annotations in a JSON file.
    Adapted from: https://stackoverflow.com/questions/70760477/cropping-a-video-to-a-specific-range-of-frames-and-region-of-interest-in-python

    :param string video_path: Path to the folder containing the video files.
    :param string json_path: Path to the JSON file.
    :param string output_dir: Path to the output folder.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)

    original_output_dir = output_dir

    for video in data:
        #Extract annotations
        annotations = video["annotations"][0]["result"]
        video_name = video["file_upload"].split("-")[1]

        video_dir = os.path.join(video_path, video_name)

        cap = cv2.VideoCapture(video_dir)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        output_dir = os.path.join(original_output_dir, video_name)
        os.makedirs(output_dir, exist_ok=True)

        clip_number = 1

        for annotation in annotations:
            label = annotation["value"]["timelinelabels"][0]
            segment_start = annotation["value"]["ranges"][0]["start"]
            segment_end = annotation["value"]["ranges"][0]["end"]


            for frame_start in range(segment_start, segment_end + 1, 30):
                clip_start = frame_start
                clip_end = min(frame_start + 30, segment_end+1)

                #Set video reader to the start frame of the clip
                cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)

                if clip_end == segment_end+1:
                    output_path = os.path.join(
                        output_dir,
                        f"SHORT_CLIP_{video_name}_{label}_{clip_start}-{clip_end-1}_clip_{clip_number}.mp4"
                    )
                else:
                    output_path = os.path.join(
                        output_dir,
                        f"{video_name}_{label}_{clip_start}-{clip_end-1}_clip_{clip_number}.mp4"
                    )
                clip_number += 1


                #Write frames to the clip
                out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
                for frame_num in range(clip_start, clip_end):
                    ret, frame = cap.read()

                    if not ret:
                        print(f"End of video reached or error at frame {frame_num}.")
                        break
                    out.write(frame)

                out.release()
                print(f"Segment saved: {output_path}")


        cap.release()
        print("Processing complete.")




video_file = "REPLACE WITH VIDEO DIRECTORY"
json_file = "REPLACE WITH JSON ANNOTATIONS DIRECTORY"
output_folder = "REPLACE WITH OUTPUT DIRECTORY"
cut_video_by_annotations(video_file, json_file, output_folder)
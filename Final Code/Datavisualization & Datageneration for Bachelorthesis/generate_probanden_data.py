import os

active_folder = 'DIRECTORY TO ACTIVE DATA'
idle_folder = 'DIRECTORY TO IDLE DATA'
videos_file_path = 'DIRECTORY TO FILE OF VIDEO NAMES'


def count_videos(name, videos):
    """
    Counts how many times a given name appears in a list of video filenames.

    :param string name: Participant name.
    :param list[string] videos: List of video filenames.
    :return int: Returns the number of times a given name appears in a list of video filenames.
    """
    return sum(1 for video in videos if name in video)


#Reads directory data
active_videos = os.listdir(active_folder)
idle_videos = os.listdir(idle_folder)


with open(videos_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()


training = True
names_training, names_val_test = [], []
for l in lines:
    if l.startswith('Train') or l.startswith('Val'):
        continue
    elif len(l.strip()) == 0:
        training = False
    else:
        if training:
            names_training.append(l.strip())
        else:
            names_val_test.append(l.strip())


#Testperson that was handled separately
specialname_entries = [line.strip() for line in lines if line.startswith('SPECIAL_NAME')]


results = []

#Collect and count videos
for name in names_training:
    active_count = count_videos(name, active_videos)
    idle_count = count_videos(name, idle_videos)
    total_count = active_count + idle_count
    results.append((name, active_count, idle_count, total_count))

results.append(("Val", 0, 0, 0))
for name in names_val_test:
    active_count = count_videos(name, active_videos)
    idle_count = count_videos(name, idle_videos)
    total_count = active_count + idle_count
    results.append((name, active_count, idle_count, total_count))

#Format the Data
print(f'{"Name":<40} {"Active":<10} {"Idle":<10} {"Total":<10}')
print('-' * 70)
print("TRAIN SPLIT")
print('-' * 70)
total = 0
total_active = 0
total_idle = 0
total_total = 0
total_total_active = 0
total_total_idle = 0


for name, active_count, idle_count, total_count in results:
    if name == 'Val':
        print('-' * 70)
        print(f"{"SUM_TOTAL"}{total_active:>36} {total_idle:>10} {total:>11}")
        print('-' * 70)
        print("VALIDATION/TEST SPLIT")
        print('-' * 70)
        total_total_active += total_active
        total_total_idle += total_total_idle
        total_total += total
        total = 0
        total_active = 0
        total_idle = 0
        continue
    total+= total_count
    total_active += active_count
    total_idle += idle_count
    print(f'{name:<40} {active_count:<10} {idle_count:<10} {total_count:<10}')


print('-' * 70)
print(f"{"SUM_TOTAL"}{total_active:>36} {total_idle:>10} {total:>10}")
print('-' * 70)
total_total_active += total_active
total_total_idle += total_total_idle
total_total += total
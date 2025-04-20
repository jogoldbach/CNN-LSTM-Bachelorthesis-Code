import os

test_dir = 'DIRECTORY TO TEST DATA'
train_dir = 'DIRECTORY TO TRAIN DATA'
videos_file_path = 'DIRECTORY TO FILE OF VIDEO NAMES'


def count_videos(name, videos):
    """
    Counts how many times a given name appears in a list of video filenames.

    :param string name: Participant name.
    :param list[string] videos: List of video filenames.
    :return int: Returns the number of times a given name appears in a list of video filenames.
    """
    return sum(1 for video in videos if name in video)


active_test_videos = os.listdir(os.path.join(test_dir, "Active"))
idle_test_videos = os.listdir(os.path.join(test_dir, "Idle"))
test_video = active_test_videos.extend(idle_test_videos)

active_train_videos = os.listdir(os.path.join(train_dir, "Active"))
idle_train_videos = os.listdir(os.path.join(train_dir, "Idle"))
train_video = active_train_videos.extend(idle_train_videos)


with open(videos_file_path, 'r') as file:
    lines = file.readlines()


names = []
for l in lines:
    names.append(l.strip())



results = []

#Collect and count training and testing data
for name in names:
    active_count = count_videos(name, active_train_videos)
    idle_count = count_videos(name, idle_train_videos)
    total_count = active_count + idle_count
    results.append((name, active_count, idle_count, total_count))

results.append(("Test", 0, 0, 0))
for name in names:
    active_test_count = count_videos(name, active_test_videos)
    idle_test_count = count_videos(name, idle_test_videos)
    total_count = active_test_count + idle_test_count
    results.append((name, active_test_count, idle_test_count, total_count))


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
    if name == 'Test':
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
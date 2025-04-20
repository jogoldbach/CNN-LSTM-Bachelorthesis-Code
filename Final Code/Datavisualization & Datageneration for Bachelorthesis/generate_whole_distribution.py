import os

data_dir = 'DIRECTORY TO VIDEO DATA'
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
active_videos = os.listdir(os.path.join(data_dir, "Active"))
idle_videos = os.listdir(os.path.join(data_dir, "Idle"))


with open(videos_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()


#Testperson that was handled separately
specialname_entries = [line.strip() for line in lines if line.startswith('SPECIAL_NAME')]
rest_participants = [line.strip() for line in lines if not line.startswith('SPECIAL_NAME')]


results = []

#Collect and count videos
for name in rest_participants:
    active_count = count_videos(name, active_videos)
    idle_count = count_videos(name, idle_videos)
    total_count = active_count + idle_count
    results.append((name, active_count, idle_count, total_count))

results.append(("SPECIAL_NAME", 0, 0, 0))
for name in specialname_entries:
    active_count = count_videos(name, active_videos)
    idle_count = count_videos(name, idle_videos)
    total_count = active_count + idle_count
    results.append((name, active_count, idle_count, total_count))


#Format the Data
print(f'{"Name":<40} {"Active":<10} {"Idle":<10} {"Total":<10}')
print('-' * 70)
print("EXPERIMENT 1")
print('-' * 70)
total = 0
total_active = 0
total_idle = 0
total_total = 0
total_total_active = 0
total_total_idle = 0

for name, active_count, idle_count, total_count in results:
    if name == 'SPECIAL_NAME':
        print('-' * 70)
        print(f"{"SUM_TOTAL"}{total_active:>36} {total_idle:>10} {total:>11}")
        print(f"{"SUM_PORTION"}{((total_active / total) * 100):>35.2f}% {(total_idle / total) * 100:>9.2f}% {(total / total) * 100:>10.2f}%")
        print('-' * 70)
        print("EXPERIMENT 2")
        print('-' * 70)
        total_total_active += total_active
        total_total_idle += total_idle
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
print(f"{"SUM_PORTION"}{((total_active/total)*100):>35.2f}% {(total_idle/total)*100:>9.2f}% {(total/total)*100:>9.2f}%")
print('-' * 70)
total_total_active += total_active
total_total_idle += total_idle
total_total += total
print("BOTH EXPERIMENTS TOTAL")
print('-' * 70)
print(f"{"SUM_TOTAL"}{total_total_active:>36} {total_total_idle:>10} {total_total:>10}")
print(f"{"SUM_PORTION"}{((total_total_active/total_total)*100):>35.2f}% {(total_total_idle/total_total)*100:>8.2f}% {(total_total/total_total)*100:>10.2f}%")
print('-' * 70)
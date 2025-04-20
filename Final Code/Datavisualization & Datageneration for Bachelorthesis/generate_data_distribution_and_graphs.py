import matplotlib.pyplot as plt
import numpy
import numpy as np

def values_piechart(val):
    """
    Converts a numerical value into a percentage value.

    :param float val: Value to be converted to percentage.
    :return string: Percentage value.
    """
    percentage  = numpy.round(val, 2)
    return str(percentage)+"%"


def make_pie_chart(training_active, training_idle, test_val_active, test_val_idle, titel):
    """
    Generates a pie chart showing the distribution of active and idle clips.

    :param list[int] training_active: Active clips in training set.
    :param list[int] training_idle: Idle clips in training set.
    :param list[int] test_val_active: Active clips in test/val set.
    :param list[int] test_val_idle: Idle clips in test/val set.
    :param string titel: Title of the pie chart.
    """
    plt.figure(figsize=(7, 7))
    pie_data = [sum(training_active), sum(training_idle), sum(test_val_active), sum(test_val_idle)]
    pie_lables = ["Trainingsdaten Active", "Trainingsdaten Idle", "Test-und Validierungsdaten Active",
                  "Test-und Validierungsdaten Idle"]
    pie_colors = ["seagreen", "lightgreen", "royalblue", "lightblue"]

    plt.pie(pie_data, colors=pie_colors, autopct=values_piechart)
    plt.legend(labels=pie_lables)
    plt.suptitle(titel)
    plt.show()

def make_bar_graph(training_active, training_idle, training_video_names, test_val_active, test_val_idle, test_val_video_names, title_graph_1, title_graph_2, y_lim):
    """
    Generates two bar graphs showing the distribution of active and idle clips in training and test sets.

    :param list[int] training_active: Active clips in training set.
    :param list[int] training_idle: Idle clips in training set.
    :param list[int] training_video_names: Names of videos in training set.
    :param list[int] test_val_active: Active clips in test/val set.
    :param list[int] test_val_idle: Idle clips in test/val set.
    :param list[int] test_val_video_names: Names of videos in test/val set.
    :param string title_graph_1: Title of graph 1.
    :param string title_graph_2: Title of graph 2.
    :param int y_lim: Maximum value for y-axis scaling.
    """

    figure, axis = plt.subplots(1, 2, figsize=(20, 8))

    x1 = np.arange(len(training_video_names))


    axis[0].bar(x1, training_active, color='seagreen', label='Active')
    axis[0].bar(x1, training_idle, bottom=training_active, color='lightgreen', label='Idle')
    axis[0].set_xticks(x1)
    axis[0].set_xticklabels(training_video_names, rotation=45, ha="right")
    axis[0].set_ylim(1,y_lim+100)
    axis[0].set_ylabel('Anzahl an Clips')
    axis[0].set_title(title_graph_1)
    axis[0].legend()


    x2 = np.arange(len(test_val_video_names))

    axis[1].bar(x2, test_val_active, color='royalblue', label='Active')
    axis[1].bar(x2, test_val_idle, bottom=test_val_active, color='lightblue', label='Idle')
    axis[1].set_xticks(x2)
    axis[1].set_xticklabels(test_val_video_names, rotation=45, ha="right")
    axis[1].set_ylim(1,y_lim+100)
    axis[1].set_ylabel('Anzahl an Clips')
    axis[1].set_title(title_graph_2)
    axis[1].legend()

    plt.tight_layout()
    plt.show()


def basic_data_distribution(basic_data):
    """
    Processes and visualizes the complete data clip distribution by experiments.

    :param list[string] basic_data: Data of the basic distribution.
    """
    experiement1_video_names = []
    experiement2_video_names = []

    experiement1_active = []
    experiement2_active = []

    experiement1_idle = []
    experiement2_idle = []

    max_per_participant=0

    switch = True

    participant_number = 1
    video_number = 1
    for line in basic_data:
        current_line = line.split()
        if len(current_line)==0 or current_line[0].startswith("-"):
            continue

        #Find switch between Train and Val Data
        if len(current_line) == 2:
            switch = not switch
            continue

        if not switch:
            experiement1_video_names.append(f"Participant {participant_number}")
            participant_number+=1
            experiement1_active.append(int(current_line[1]))
            experiement1_idle.append(int(current_line[2]))
            if max_per_participant < (int(current_line[1])+int(current_line[2])):
                max_per_participant = int(current_line[1])+int(current_line[2])
        else:
            experiement2_video_names.append(f"Participant {participant_number} Video {video_number}" )
            video_number+=1
            experiement2_active.append(int(current_line[1]))
            experiement2_idle.append(int(current_line[2]))
            if max_per_participant < (int(current_line[1])+int(current_line[2])):
                max_per_participant = int(current_line[1])+int(current_line[2])

    make_bar_graph(experiement1_active, experiement1_idle, experiement1_video_names, experiement2_active, experiement2_idle, experiement2_video_names, "Verteilung Balkendiagramm - Experiment 1", "Verteilung Balkendiagramm - Experiment 2", max_per_participant)


#Manual datasplit data: By video
def manual_data(manual_data):
    """
    Processes and visualizes the data clip distribution of the manual dataset.

    :param list[string] manual_data: Data of the manual dataset.
    """
    switch = True

    training_video_names = []
    test_val_video_names = []

    training_active = []
    test_val_active = []

    training_idle = []
    test_val_idle = []

    max_per_participant = 0

    participant_number = 1
    video_number = 1
    for line in manual_data:
        current_line = line.split()
        #Filter non interesting data
        if len(current_line)==0 or current_line[0].startswith("-"):
            continue
        #Find switch between Train and Val Data
        if len(current_line) == 2:
            switch = not switch
            continue

        if not switch:
            if current_line[0].startswith("NAME_OF_PARTICIPANT"):
                training_video_names.append(f"Participant 11 Video {video_number}")
                video_number += 1
                training_active.append(int(current_line[1]))
                training_idle.append(int(current_line[2]))
                if max_per_participant < (int(current_line[1]) + int(current_line[2])):
                    max_per_participant = int(current_line[1]) + int(current_line[2])
            else:
                training_video_names.append(f"Participant {participant_number}")
                participant_number += 1
                training_active.append(int(current_line[1]))
                training_idle.append(int(current_line[2]))
                if max_per_participant < (int(current_line[1]) + int(current_line[2])):
                    max_per_participant = int(current_line[1]) + int(current_line[2])
        else:
            if current_line[0].startswith("NAME_OF_PARTICIPANT"):
                test_val_video_names.append(f"Participant 11 Video {video_number}")
                video_number += 1
                test_val_active.append(int(current_line[1]))
                test_val_idle.append(int(current_line[2]))
                if max_per_participant < (int(current_line[1]) + int(current_line[2])):
                    max_per_participant = int(current_line[1]) + int(current_line[2])
            else:
                test_val_video_names.append(f"Participant {participant_number}")
                participant_number += 1
                test_val_active.append(int(current_line[1]))
                test_val_idle.append(int(current_line[2]))
                if max_per_participant < (int(current_line[1]) + int(current_line[2])):
                    max_per_participant = int(current_line[1]) + int(current_line[2])



    make_pie_chart(training_active, training_idle, test_val_active, test_val_idle, "Verteilung Manueller Datensplit")

    make_bar_graph(training_active, training_idle, training_video_names, test_val_active, test_val_idle, test_val_video_names, "Verteilung nach Teilnehmer Balkendiagramm - Trainingsdaten", "Verteilung nach Teilnehmer Balkendiagramm - Testdaten", max_per_participant)



#Random datasplit data
def random_data(random_names, random_data):
    """
    Processes and visualizes the data clip distribution of the random dataset.

    :param list[string] random_names: List of participant names to anonymize.
    :param list[string] random_data: Data of the random dataset.
    """
    name_dict = {}

    video_number = 1
    participant_number = 1

    for name in random_names:
        name = name.strip()
        if name.startswith("NAME_OF_PARTICIPANT"):
            name_dict[name] = f"Participant 11 Video {video_number}"
            video_number+=1
        else:
            name_dict[name]=f"Participant {participant_number}"
            participant_number+=1

    switch = True

    training_video_names = []
    test_val_video_names = []

    training_active = []
    test_val_active = []

    training_idle = []
    test_val_idle = []

    max_per_participant = 0

    participant_number = 1

    for line in random_data:
        current_line = line.split()
        #Filter non interesting data
        if len(current_line)==0 or current_line[0].startswith("-"):
            continue
        #Find switch between Train and Val Data
        if len(current_line) == 2:
            switch = not switch
            continue
        if not switch:
            training_video_names.append(name_dict[current_line[0].strip()])
            participant_number += 1
            training_active.append(int(current_line[1]))
            training_idle.append(int(current_line[2]))
            if max_per_participant < (int(current_line[1])+int(current_line[2])):
                max_per_participant = int(current_line[1])+int(current_line[2])
        else:
            test_val_video_names.append(name_dict[current_line[0].strip()])
            participant_number += 1
            test_val_active.append(int(current_line[1]))
            test_val_idle.append(int(current_line[2]))
            if max_per_participant < (int(current_line[1])+int(current_line[2])):
                max_per_participant = int(current_line[1])+int(current_line[2])


    make_pie_chart(training_active, training_idle, test_val_active, test_val_idle, "Verteilung zufälliger Datensplit")


    make_bar_graph(training_active, training_idle, training_video_names, test_val_active, test_val_idle, test_val_video_names, "Verteilung nach Zufall Balkendiagramm - Trainingsdaten", "Verteilung nach Zufall Balkendiagramm - Testdaten", max_per_participant)



random_selection_data_distribution = "DIRECTORY FOR DATADISTRIBUTION OF RANDOM DATASPLIT"
manual_participant_data_distribution =  "DIRECTORY FOR DATADISTRIBUTION OF MANUAL DATASPLIT"
all_data_distribution = "DIRECTORY FOR DATADISTRIBUTION OF ALL VIDEOS"
video_types = "DIRECTORY FOR LIST OF ALL VIDEO NAMES"


with open(random_selection_data_distribution, 'r') as random_file:
    lines_random = random_file.readlines()

with open(manual_participant_data_distribution, 'r') as manual_file:
    lines_manual = manual_file.readlines()

with open(all_data_distribution, 'r') as whole_file:
    lines_whole = whole_file.readlines()

with open(video_types, 'r') as video_names_file:
    lines_names = video_names_file.readlines()

random_data(lines_names, lines_random)
manual_data(lines_manual)
basic_data_distribution(lines_whole)

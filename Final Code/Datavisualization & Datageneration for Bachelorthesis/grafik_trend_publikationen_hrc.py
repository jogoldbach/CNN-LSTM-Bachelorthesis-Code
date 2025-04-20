import matplotlib.pyplot as plt

"""
This script was used to make a graph for the HRC trend.

"""

#Bar graph data
categories = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022','2023', '2024',]
values = [ 206, 292, 412, 545, 646, 793, 940, 1099, 1310, 2245 ]

plt.figure(figsize=(8, 6))
bars = plt.bar(categories, values, color='lightblue')
plt.xlabel('Jahre')
plt.ylabel('Publikationen mit dem Thema MRK')
plt.title('Publikations Trend MRK')

#Add exact numbers on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
             f'{value}', ha='center', va='bottom', fontsize=10)


plt.tight_layout()
plt.show()
from settings.soc_struct import *
from settings.socs_to_test import *
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString

import sys

if __name__ == "__main__":

  scaling_arg = 'high_margin'
  if len(sys.argv) > 1:
    if sys.argv[1] == 'naive' or sys.argv[1] == 'high_margin':
      scaling_arg = sys.argv[1]

  #Print into a log file
  original_stdout = sys.stdout
  log_file = open("logs/log_ook_show.txt", "w")
  sys.stdout = log_file #print into log file

  #parameters
  fontscale = 1
  channel_density = 1

  #change to get different plots
  if scaling_arg == 'naive':
    antenna_density = 1
  else:
    antenna_density = 0

  extra_sensing_power = 1
  extra_non_sensing_power = 1

  max_ch = (8192 + 1024 + 8192)*2
  plot_name = scaling_arg

  ctr = 0

  width = 10
  height = 2.5

  for soc in socs:
    ctr = ctr + 1

    name = soc[0]["Name"]

    data = np.loadtxt(f'data/ook_data{name}_{antenna_density}.txt')
    ni_channels, sensing_power_consumption, total_power_budget, non_sensing_power_consumption, total_power_consumption, sensing_area, total_area = data.T

    ni_channels = ni_channels.tolist()

    bar_width = 0.1
    categories = ['1024', '2048', '4096', '8192']#, '16384', '32768']

    x = np.arange(len(categories))

    sensing_list = list()
    non_sensing_list = list()
    total_list = list()
    budget_list = list()

    for i in range(len(categories)):
      idx = ni_channels.index(1024*2**i)
      sensing_list = sensing_list + [sensing_power_consumption[idx] / total_power_budget[idx]]
      non_sensing_list = non_sensing_list + [non_sensing_power_consumption[idx] / total_power_budget[idx]]
      total_list = total_list + [total_power_consumption[idx] / total_power_budget[idx]]
      budget_list = budget_list + [1]

    plt.figure(1,figsize=(10, 3.5))
    if soc[0]["Name"] == '1':
      bars = plt.bar(x - (4-ctr)*bar_width+ctr*0.005, total_list, width=bar_width, label='Non-Sensing', color='blue')
      plt.bar(x - (4-ctr)*bar_width+ctr*0.005, sensing_list, width=bar_width, label='Sensing', color='green')
    else:
      bars = plt.bar(x - (4-ctr)*bar_width+ctr*0.005, total_list, width=bar_width, color='blue')
      plt.bar(x - (4-ctr)*bar_width+ctr*0.005, sensing_list, width=bar_width, color='green')

    for bar in bars:
      height = bar.get_height()
      name = soc[0]["Name"]

      if antenna_density == 0:
        plt.text(bar.get_x() + bar.get_width() / 2, height - 0.4, f'{name}', ha='center', va='bottom', fontsize=16*fontscale, color='white')
      else:
        plt.text(bar.get_x() + bar.get_width() / 2, height - 0.15, f'{name}', ha='center', va='bottom', fontsize=16*fontscale, color='white')

    if antenna_density == 0:
      plt.figure(2, figsize=(7, 5)) #area/w
    else:
      plt.figure(2, figsize=(5.9, 5)) #area/w
    start_idx = ni_channels.index(1024)
    normalized_sensing_area = np.divide(sensing_area[start_idx:], total_area[start_idx:])
    plt.plot(ni_channels[start_idx:], normalized_sensing_area, linewidth=3, label=f'{name}', color=colors[ctr-1])


  plt.figure(1)
  plt.axhline(y=1, color='red', linestyle='--', linewidth=3, label='Power Budget')
  plt.xlabel('Number of NI Channels', fontsize=18*fontscale)
  plt.ylabel('Relative Power', fontsize=18*fontscale)
  plt.xticks(x, categories, fontsize=16*fontscale)
  plt.yticks(fontsize=16*fontscale)
  plt.legend(fontsize=16*fontscale,loc='upper center', bbox_to_anchor=(0.7, 1.2), ncol=4,columnspacing=0.5, frameon=False,  handlelength=1)

  if antenna_density == 0:
    plt.text(
      0.005, 1.11,
      "High-Margin Design",   # The label text
      fontsize=16,        # Text size
      weight='bold',
      transform=plt.gca().transAxes,  # Use axis-relative coordinates
      verticalalignment='top',       # Align text to the top
      horizontalalignment='left'     # Align text to the left
    )
    plt.ylim(0.0,3.5)
  else:
    plt.text(
      0.005, 1.11,
      "Naive Design",   # The label text
      fontsize=16,        # Text size
      weight='bold',
      transform=plt.gca().transAxes,  # Use axis-relative coordinates
      verticalalignment='top',       # Align text to the top
      horizontalalignment='left'     # Align text to the left
    )

  plt.tight_layout()
  plt.savefig(f"figures/{plot_name}_comm_scaling_socs.pdf", transparent=True)

  plt.figure(2)
  plt.xlabel('Number of NI Channels', fontsize=22*fontscale)
  plt.ylabel('Relative Sensing Area', fontsize=22*fontscale)
  plt.xticks(ni_channels[::32], fontsize=16*fontscale)
  plt.yticks(fontsize=16*fontscale)

  plt.ylim(0.0, 1.0)
  plt.xlim(1024,8192)

  if antenna_density == 0:
    plt.text(
      0.005, 1.07,
      "High-Margin Design",   # The label text
      fontsize=20,        # Text size
      weight='bold',
      transform=plt.gca().transAxes,  # Use axis-relative coordinates
      verticalalignment='top',       # Align text to the top
      horizontalalignment='left'     # Align text to the left
    )

    plt.legend(fontsize=20*fontscale,loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1,columnspacing=0.5, frameon=False)

  else:
    plt.text(
      0.005, 1.07,
      "Naive Design",   # The label text
      fontsize=20,        # Text size
      weight='bold',
      transform=plt.gca().transAxes,  # Use axis-relative coordinates
      verticalalignment='top',       # Align text to the top
      horizontalalignment='left'     # Align text to the left
    )


  ax = plt.gca()  # get current axes

  # Move x-tick labels downward
  for label in ax.get_xticklabels():
    label.set_y(label.get_position()[1] - 0.01)  # tweak the value as needed

  plt.tight_layout()
  plt.savefig(f"figures/{plot_name}_area_scaling_socs.pdf", transparent=True)

  plt.show()

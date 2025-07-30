from scaling.scaling_qam_comm import *
from settings.soc_struct import *
from settings.socs_to_test import *

import sys

if __name__ == "__main__":

  #Print into a log file
  original_stdout = sys.stdout
  log_file = open("logs/log_qam_show.txt", "w")

  #parameters
  qam_efficiency = 0
  qam_max = 99
  qam_min = 1
  plot_name = 'qam'
  step = 128

  max_ch = 8192

  channel_list = list()
  ctr = 0

  efficiency_20_list = list()
  channels_efficiency_list = list()
  eff_avg = 99

  for i in range(9216):
    channel_list.append(i)

  fig, ax = plt.subplots(figsize=(13, 5))

  for soc in socs:

    sys.stdout = log_file #print into log file

    ctr = ctr + 1

    name = soc[0]["Name"]

    data = np.loadtxt(f'data/qam_data{name}.txt')
    channel_accurate_list, efficiency_accurate_list = data.T  # Transpose to unpack columns

    plt.plot(channel_accurate_list, efficiency_accurate_list, linestyle='-', label=f'{name}', linewidth=2, color=colors[ctr-1])

    for i in range(len(efficiency_accurate_list)):
      if efficiency_accurate_list[i] != 0:
        print("efficiency ", efficiency_accurate_list[i])
        print("channels ", channel_accurate_list[i])
        if efficiency_accurate_list[i] == 20:
          efficiency_20_list.append(channel_accurate_list[i])
        if efficiency_accurate_list[i] == eff_avg:
          channels_efficiency_list.append(channel_accurate_list[i])


  print(len(efficiency_20_list))
  average = sum(efficiency_20_list) / len(efficiency_20_list)
  print(average)
  print(len(channels_efficiency_list))
  average = sum(channels_efficiency_list) / len(channels_efficiency_list)
  print(average)

  plt.xlabel('Number of NI Channels', fontsize=24)
  plt.ylabel('QAM Efficiency [%]', fontsize=24)
  plt.xticks(channel_list[::step*8], fontsize=22)
  plt.yticks(fontsize=22)
  plt.legend(fontsize=28,loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=8,columnspacing=0.5, frameon=False, handlelength=0.8)
  plt.grid()
  plt.xlim(1024, max_ch-2048)
  plt.tight_layout()
  plt.savefig(f"figures/{plot_name}_comm_scaling_socs.pdf", transparent=True)
  plt.show()

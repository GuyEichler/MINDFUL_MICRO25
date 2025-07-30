from scaling.scaling_dnn_comp import *
from settings.soc_struct import *
from settings.socs_to_test import *

import sys

if __name__ == "__main__":

  #Print into a log file
  original_stdout = sys.stdout
  log_file = open("logs/log_dnn_show.txt", "w")

  #parameters
  dnn_types = ["mlp", "dense"]
  step = 32
  max_ch = 8192+1024+1024+1024

  ctr = 0

  width = 8
  height = 5

  for soc in socs:
    ctr = ctr + 1

    for i in range(len(dnn_types)):
      sys.stdout = log_file #print into log file

      #Choose a DNN architecture to run
      plot_name = dnn_types[i]
      if dnn_types[i] == "mlp":
        dnn_arch = mlp_architecture.copy()
      elif dnn_types[i] == "dense":
        dnn_arch = densenet_architecture.copy()
      else:
        dnn_arch = densenet_architecture.copy()

      name = soc[0]["Name"]

      data = np.loadtxt(f'data/{plot_name}_comp_data{name}.txt')
      total_power_budget, total_power_consumption, ni_channels = data.T

      ni_channels = ni_channels.tolist()

      start_idx = ni_channels.index(1024)
      normalized_leftover_power = np.subtract(total_power_budget[start_idx:], total_power_consumption[start_idx:])
      normalized_leftover_power = np.divide(normalized_leftover_power, total_power_budget[start_idx:])

      normalized_total_power = np.subtract(total_power_budget[start_idx:], total_power_consumption[start_idx:])
      normalized_total_power = np.divide(total_power_consumption[start_idx:], total_power_budget[start_idx:])

      if i == 0:
        plt.figure(i+1, figsize=(7.5,5))
      else:
        plt.figure(i+1, figsize=(5.6,5))
      name = soc[0]["Name"]
      normalized_leftover_power[0] = 0

      if len(normalized_leftover_power) > 1:
        plt.plot(ni_channels[start_idx+1:], normalized_total_power[1:], linewidth=3, label=f'{name}', color=colors[ctr-1])



  plt.figure(1)
  plt.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Power\nBudget')
  plt.xlabel('Number of NI Channels', fontsize=18)
  plt.ylabel('Normalized Power', fontsize=22)

  xticks = np.arange(0, max_ch + 1, step * step)
  plt.xticks(xticks, fontsize=16)

  #plt.xticks(ni_channels[start_idx::step*scale_x], fontsize=16)
  plt.yticks(fontsize=16)

  plt.xlim(1024, max_ch-4096)
  plt.legend(fontsize=20,loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1,columnspacing=0.5, frameon=False, handlelength=0.8)
  plt.grid()

  plt.text(
    0.98, 0.98,                # X and Y coordinates (normalized to axes)
    "MLP",         # Text to display
    fontsize=16,               # Font size
    ha='right', va='top',      # Align horizontally to the right and vertically to the top
    transform=plt.gca().transAxes,  # Use axis coordinates (0 to 1)
    fontweight='bold'          # Make the text bold (optional)
  )

  plt.tight_layout()
  plt.savefig(f"figures/mlp_comp_scaling_socs.pdf", transparent=True)


  plt.figure(2)
  plt.axhline(y=1, color='red', linestyle='--', linewidth=2, label="Power\nBudget")
  plt.xlabel('Number of NI Channels', fontsize=18)
  plt.ylabel('Normalized Power', fontsize=22)

  xticks = np.arange(0, max_ch, step * step)
  plt.xticks(xticks, fontsize=16)

  plt.yticks(fontsize=16)
  plt.xlim(1024, max_ch-4096)

  plt.grid()

  plt.text(
    0.98, 0.98,                # X and Y coordinates (normalized to axes)
    "DN-CNN",         # Text to display
    fontsize=16,               # Font size
    ha='right', va='top',      # Align horizontally to the right and vertically to the top
    transform=plt.gca().transAxes,  # Use axis coordinates (0 to 1)
    fontweight='bold'          # Make the text bold (optional)
  )

  plt.tight_layout()
  plt.savefig(f"figures/dense_comp_scaling_socs.pdf", transparent=True)

  plt.show()

from framework.networks import *
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import differential_evolution

def scaling_qam_comm(soc, physical_specs, comm_specs, show=True, additional_max_channels=0, step=4, maximum_channels=2624):

  physical = physical_specs.copy()
  comm = comm_specs.copy()

  #max_channels = (2048+64+512)+additional_max_channels
  max_channels = maximum_channels
  min_channels = 0
  step = step

  orig_channels = soc["active_channels"]
  max_orig_channels = soc["max_channels"]
  sensing_area = soc["sensing_area"]
  total_area = soc["total_area"]
  non_sensing_area = total_area - sensing_area
  power_consumption = soc["power_consumption"]
  max_comm_channels = soc["max_comm_channels"]

  #max_data_rate = 300 #Mb/sec
  data_type = soc["data_type"]
  network_time = soc["sampling_period"]
  #max_comm_channels = 300*10**6 / data_type * network_time * 10**-9
  power_budget_per_area = soc["power_density_budget"]

  non_sensing_area_per_channel = non_sensing_area / max_comm_channels
  sensing_area_per_channel = sensing_area / orig_channels #0.000768 #mm^2
  area_per_channel = total_area / orig_channels

  #power_per_channel_recording = 0.01325 #mW
  power_per_channel_recording = power_consumption * sensing_area/total_area / orig_channels
  power_per_channel_comm = power_consumption * non_sensing_area/total_area / max_comm_channels
  #power_per_channel = 0.03789 #mW

  name = soc["Name"]
  print(f"{name}: Non sensing area per channel:", non_sensing_area_per_channel)
  print(f"{name}: Sensing area per channel:", sensing_area_per_channel)
  print(f"{name}: Max communication channels:", max_comm_channels)


  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  total_power_budget_plot = list()
  sensing_power_budget_plot = list()
  non_sensing_power_budget_plot = list()
  sensing_power_consumption_plot = list()
  non_sensing_power_consumption_plot = list()
  power_consumption_plot = list()

  #tick_ratio = math.ceil(max_channels / 256 * 16 / step)
  tick_ratio = 64

  for channels in channel_numbers:
    print("Num channels:", channels)

    if channels <= orig_channels:
      #area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel
      if channels == 0:
        total_power_budget_plot.append(0)
        sensing_power_consumption_plot.append(0)
        non_sensing_power_consumption_plot.append(0)
        power_consumption_plot.append(0)

        #Prepare x axis - channel number
        x_val = channels
        x_axis.append(x_val)
        continue

      total_area = area_per_channel * channels
      total_power_budget = power_budget_per_area * total_area

      sensing_power_consumption = power_per_channel_recording * channels
      non_sensing_power_consumption = power_per_channel_comm * channels
      power_consumption_channels = (power_per_channel_recording + power_per_channel_comm) * channels

      # print("\nPOWER BUDGET:", new_power_budget, "mW")
      # print("TOTAL AREA:", total_area, "mm^2")
      # print("SENSING AREA:", total_sensing_area, "mm^2")
      # print("NON SENSING AREA:", total_non_sensing_area, "mm^2\n")

      total_power_budget_plot.append(total_power_budget)
      sensing_power_consumption_plot.append(sensing_power_consumption)
      non_sensing_power_consumption_plot.append(non_sensing_power_consumption)
      power_consumption_plot.append(power_consumption_channels)

    else: #communication scaling

      #sensing power and area
      sensing_power_consumption = power_per_channel_recording * channels
      sensing_area = sensing_area_per_channel * channels

      #non-sensing power - communication
      #calculate the communication - no QAM
      physical["network_time"] = network_time
      physical["data_type"] = data_type
      communication_power_orig, data_rate_orig = calc_communication(orig_channels, 0, physical, comm, enable_qam=False)

      comm["max_data_rate"] = data_rate_orig / 10**6
      communication_power_qam, data_rate_qam = calc_communication(channels, 0, physical, comm, enable_qam=True)

      #Do have accurate power - remove the power from the original transmission
      non_sensing_power_consumption = \
        power_per_channel_comm * orig_channels - communication_power_orig * 10**3 + \
        communication_power_qam * 10**3

      #power budget
      total_power_budget = power_budget_per_area * (sensing_area + non_sensing_area)

      #power consumption
      power_consumption_channels = sensing_power_consumption + non_sensing_power_consumption

      total_power_budget_plot.append(total_power_budget) #
      sensing_power_consumption_plot.append(sensing_power_consumption) #
      non_sensing_power_consumption_plot.append(non_sensing_power_consumption)
      power_consumption_plot.append(power_consumption_channels)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)

  #intersection
  y1 = np.array(power_consumption_plot)
  y2 = np.array(total_power_budget_plot)
  x = x_axis

  x_intersections, y_intersections = intersection_finder(y1, y2, x)

  if show == True:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axvline(x=1024, color='b', linestyle=':', linewidth=2, label="Data rate limit")
    # label_xx, label_yy = 1024-40, total_power_budget_plot[mid_index * 2 - 10]

    x = np.array(x_axis)
    y = np.array(total_power_budget_plot)
    offset_angle = int(tick_ratio/step - step+1)
    offset_v = int(tick_ratio/step - step*2 - 2)

    x_label, y_label, angle = label_position(x , y, offset_angle, offset_v)

    plt.plot(x_axis, total_power_budget_plot, linestyle='--', linewidth=4, color='red', label=None)
    plt.text(x_label, y_label, 'Power Budget', rotation=angle, ha='center', va='center', color='red', weight='bold', fontsize=18)



    plt.plot(x_axis, sensing_power_consumption_plot, linewidth=3, label='Sensing Power')
    plt.plot(x_axis, non_sensing_power_consumption_plot, linewidth=3, label='Non-Sensing Power')
    plt.plot(x_axis, power_consumption_plot, linewidth=3, label='Total Power Consumption')
    plt.xticks(x_axis[::tick_ratio], fontsize=18)
    plt.yticks(fontsize=18)

    plt.ylim(0, plt.ylim()[1]+20)
    plt.scatter(x_intersections, y_intersections, color='blue', zorder=5, s=20)
    for x_int, y_int in zip(x_intersections, y_intersections):
      plt.axvline(x=x_int, ymin=0, ymax=y_int/plt.ylim()[1], color='gray', linestyle='dashed')

    #print('y lim: ', plt.ylim()[1])

    plt.xlabel('Number of Active Channels', fontsize=22)
    plt.ylabel('Power [mW]', fontsize=22)
    plt.grid(axis='y')
    #plt.title('Number of Active Channels vs. Power')
    plt.legend(fontsize=15,loc='upper left')

    plt.tight_layout()

    plt.savefig(f"figures/{name}_qam_comm_scaling.pdf")

    plt.show()

  return x_intersections, y_intersections, x_axis, sensing_power_consumption_plot, non_sensing_power_consumption_plot, power_consumption_plot, total_power_budget_plot

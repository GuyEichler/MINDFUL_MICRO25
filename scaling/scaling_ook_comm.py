from framework.networks import *
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import differential_evolution

def scaling_ook_comm(soc, physical_specs, comm_specs, channel_density=1, antenna_density=1, extra_sensing_power=1, extra_non_sensing_power=1, show=True, maximum_channels=2624):

  physical = physical_specs.copy()
  comm = comm_specs.copy()

  rate = 1
  max_channels = maximum_channels
  min_channels = 0
  step = int(32 / rate)

  orig_channels = soc["active_channels"]
  max_orig_channels = soc["max_channels"]
  sensing_area = soc["sensing_area"]
  total_area = soc["total_area"]
  non_sensing_area = total_area - sensing_area
  power_consumption = soc["power_consumption"]
  max_comm_channels = soc["max_comm_channels"]

  # #IBIS parameters
  # ibis_orig_channels = 192 * 245
  # ibis_max_orig_channels = 192 * 245
  # ibis_sensing_area = 5.76 * 7.68 #mm^2
  # ibis_total_area = 144 #mm^2 - BISC
  # ibis_non_sensing_area = ibis_total_area - ibis_sensing_area
  # ibis_power_consumption = 48 #mW
  # ibis_max_comm_channels = ibis_orig_channels

  #max_data_rate = 300 #Mb/sec
  data_type = soc["data_type"]
  network_time = soc["sampling_period"]
  #max_comm_channels = 300*10**6 / data_type * network_time * 10**-9
  power_budget_per_area = soc["power_density_budget"]

  non_sensing_area_per_channel = non_sensing_area / max_comm_channels
  sensing_area_per_channel = sensing_area / orig_channels #0.000768 #mm^2
  area_per_channel = total_area / orig_channels

  power_per_channel_recording = power_consumption * sensing_area/total_area/orig_channels
  power_per_channel_non_sensing = power_consumption * non_sensing_area/total_area/max_comm_channels

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
  sensing_area_plot = list()
  non_sensing_area_plot = list()
  total_area_plot = list()

  #tick_ratio = math.ceil(max_channels / 256 * 16 / step)
  tick_ratio = 8 * rate

  for channels in channel_numbers:
    #print("Num channels:", channels)

    if channels <= orig_channels:
      #area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel
      if channels == 0:
        total_power_budget_plot.append(0)
        sensing_power_budget_plot.append(0)
        non_sensing_power_budget_plot.append(0)
        sensing_power_consumption_plot.append(0)
        non_sensing_power_consumption_plot.append(0)
        power_consumption_plot.append(0)
        sensing_area_plot.append(0)
        non_sensing_area_plot.append(0)
        total_area_plot.append(0)

        #Prepare x axis - channel number
        x_val = channels
        x_axis.append(x_val)
        continue

      total_area = area_per_channel * channels
      total_power_budget = power_budget_per_area * total_area
      total_sensing_area = sensing_area_per_channel * channels
      total_non_sensing_area = non_sensing_area_per_channel * channels

      sensing_power_budget = total_sensing_area/total_area * total_power_budget
      non_sensing_power_budget = total_non_sensing_area/total_area * total_power_budget

      sensing_power_consumption = power_per_channel_recording * channels
      non_sensing_power_consumption = power_per_channel_non_sensing * channels
      power_consumption_channels = (power_per_channel_recording + power_per_channel_non_sensing) * channels

      # print("\nPOWER BUDGET:", new_power_budget, "mW")
      # print("TOTAL AREA:", total_area, "mm^2")
      # print("SENSING AREA:", total_sensing_area, "mm^2")
      # print("NON SENSING AREA:", total_non_sensing_area, "mm^2\n")

      total_power_budget_plot.append(total_power_budget)
      sensing_power_budget_plot.append(sensing_power_budget)
      non_sensing_power_budget_plot.append(non_sensing_power_budget)
      sensing_power_consumption_plot.append(sensing_power_consumption)
      non_sensing_power_consumption_plot.append(non_sensing_power_consumption)
      power_consumption_plot.append(power_consumption_channels)
      sensing_area_plot.append(total_sensing_area)
      non_sensing_area_plot.append(total_non_sensing_area)
      total_area_plot.append(total_sensing_area+total_non_sensing_area)

    else:

      new_channels = channels - orig_channels
      #sensing_power_consumption = power_per_channel_recording * channels
      sensing_power_consumption = \
        power_per_channel_recording * orig_channels + \
        power_per_channel_recording * new_channels * extra_sensing_power
      #sensing_area = sensing_area_per_channel * channels
      sensing_area = \
        sensing_area_per_channel * orig_channels + \
        new_channels * sensing_area_per_channel * channel_density
      sensing_power_budget = sensing_area * power_budget_per_area

      # non_sensing_area = non_sensing_area_per_channel * channels
      non_sensing_area = \
        non_sensing_area_per_channel * orig_channels + \
        non_sensing_area_per_channel * new_channels * antenna_density
      non_sensing_power_budget = non_sensing_area * power_budget_per_area

      ratio = (non_sensing_area_per_channel * orig_channels) / non_sensing_area

      # if antenna_density == 1:
        #non_sensing_power_consumption = power_per_channel_non_sensing * channels
      non_sensing_power_consumption = \
        power_per_channel_non_sensing * orig_channels + \
        power_per_channel_non_sensing * new_channels * extra_non_sensing_power
        # else:
        #   non_sensing_power_consumption = \
        #     ratio * (power_per_channel_non_sensing * channels)
      # elif antenna_density == 0:
      #   physical["network_time"] = network_time #8KHz
      #   physical["data_type"] = data_type #10 bit
      #   communication_power, data_rate = calc_communication(channels, 0, physical, comm, enable_qam=False)
      #   communication_power_orig, data_rate = calc_communication(orig_channels, 0, physical, comm, enable_qam=False)
      #   new_communication_power = (communication_power - communication_power_orig) * 10**3

      #   non_sensing_power_consumption = \
      #     power_per_channel_non_sensing * orig_channels + \
      #     new_communication_power

      total_power_consumption = \
        sensing_power_consumption + non_sensing_power_consumption

      total_power_budget = (sensing_area + non_sensing_area) * power_budget_per_area

      print("\nPOWER BUDGET:", total_power_budget, "mW")
      print("SENSING POWER_BUDGET:", sensing_power_budget, "mW")
      print("NON SENSING POWER BUDGET:", non_sensing_power_budget, "mW")
      print("TOTAL AREA:", sensing_area+non_sensing_area, "mm^2")
      print("SENSING AREA:", sensing_area, "mm^2")
      print("NON SENSING AREA:", non_sensing_area, "mm^2")
      print("TOTAL POWER:", sensing_power_consumption+non_sensing_power_consumption, "mW")
      print("SENSING POWER:", sensing_power_consumption, "mW")
      print("NON SENSING POWER:", non_sensing_power_consumption, "mW\n")

      total_power_budget_plot.append(total_power_budget)
      sensing_power_budget_plot.append(sensing_power_budget)
      non_sensing_power_budget_plot.append(non_sensing_power_budget)
      sensing_power_consumption_plot.append(sensing_power_consumption)
      non_sensing_power_consumption_plot.append(non_sensing_power_consumption)
      power_consumption_plot.append(total_power_consumption)
      sensing_area_plot.append(sensing_area)
      non_sensing_area_plot.append(non_sensing_area)
      total_area_plot.append(sensing_area+non_sensing_area)

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
    x = np.array(x_axis)
    y = np.array(total_power_budget_plot)
    offset_angle = tick_ratio
    offset_v = tick_ratio

    x_label, y_label, angle = label_position(x , y, offset_angle, offset_v)


    ax.axvline(x=1024, color='b', linestyle=':', linewidth=2, label="Data rate limit")

    plt.plot(x_axis, total_power_budget_plot, linestyle='--', linewidth=4, color='red', label=None)
    plt.text(x_label, y_label, 'Power Budget', rotation=angle, ha='center', va='center', color='red', weight='bold', fontsize=18)

    plt.plot(x_axis, sensing_power_consumption_plot, linewidth=3, label='Sensing Power')
    plt.plot(x_axis, non_sensing_power_consumption_plot, linewidth=3, label='Non-Sensing Power')
    plt.plot(x_axis, power_consumption_plot, linewidth=3, label='Total Power Consumption')

    # #intersection
    # y1 = np.array(power_consumption_plot)
    # y2 = np.array(total_power_budget_plot)
    # x = x_axis

    # x_intersections, y_intersections = intersection_finder(y1, y2, x)

    plt.scatter(x_intersections, y_intersections, color='blue', zorder=5, s=20)
    for x_int, y_int in zip(x_intersections, y_intersections):
      plt.axvline(x=x_int, ymin=0, ymax=y_int/plt.ylim()[1], color='gray', linestyle='dashed')

    plt.xticks(x_axis[::tick_ratio], fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Number of Active Channels', fontsize=22)
    plt.ylabel('Power [mW]', fontsize=22)
    plt.grid(axis='y')
    #plt.title('Number of Active Channels vs. Power')
    plt.legend(fontsize=15,loc='upper left')

    plt.tight_layout()

    plt.savefig(f"figures/{name}_ook_comm_scaling_{int(channel_density)}_{int(antenna_density)}.pdf")

    plt.show()

  return x_intersections, y_intersections, x_axis, sensing_power_consumption_plot, non_sensing_power_consumption_plot, power_consumption_plot, total_power_budget_plot, sensing_area_plot, non_sensing_area_plot, total_area_plot

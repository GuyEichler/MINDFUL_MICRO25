from framework.networks import *
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import differential_evolution

def scaling_layers_comp(soc, dnn_arch, physical_specs, comm_specs, show=True, maximum_channels=2368, step=4):

  ctr = 0
  scale_power = 1.5

  physical = physical_specs.copy()
  comm = comm_specs.copy()

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


  # total_area = area_per_channel * channels
  # total_power_budget = power_budget_per_area * total_area

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  total_power_budget_plot = list()
  sensing_power_budget_plot = list()
  non_sensing_power_budget_plot = list()
  sensing_power_consumption_plot = list()
  # non_sensing_power_consumption_no_pipe_plot = list()
  # non_sensing_power_consumption_pipe_plot = list()
  # non_sensing_power_consumption_dense_layers_plot = list()
  # non_sensing_power_consumption_mlp_layers_plot = list()
  non_sensing_power_consumption_layers_plot = list()
  non_sensing_power_consumption_comm_plot = list()
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
        #non_sensing_power_consumption_no_pipe_plot.append(0)
        # non_sensing_power_consumption_pipe_plot.append(None)
        # non_sensing_power_consumption_dense_layers_plot.append(None)
        # non_sensing_power_consumption_mlp_layers_plot.append(None)
        non_sensing_power_consumption_layers_plot.append(None)
        non_sensing_power_consumption_comm_plot.append(0)
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
      #non_sensing_power_consumption_no_pipe_plot.append(non_sensing_power_consumption)
      # non_sensing_power_consumption_pipe_plot.append(None) #no computation yet
      # non_sensing_power_consumption_dense_layers_plot.append(None)
      # non_sensing_power_consumption_mlp_layers_plot.append(None)
      non_sensing_power_consumption_layers_plot.append(None)
      non_sensing_power_consumption_comm_plot.append(non_sensing_power_consumption)
      power_consumption_plot.append(power_consumption_channels)

    else: #computation scaling

      #parameters
      reduction = False #doesn't need to be set true can just set aggressive to true
      aggressive = True
      budget=True # sets power budget to the maximum
      check_data_rate = True
      network_input_dependency = 1

      #sensing power and area
      sensing_power_consumption = power_per_channel_recording * channels
      sensing_area = sensing_area_per_channel * channels

      #non sensing area
      #non_sensing_area - stays constant with 1024 channels
      #non_sensing_area = non_sensing_area_per_channel * channels

      #non-sensing power - computation
      #calculate the communication - no QAM
      physical["network_time"] = network_time #8KHz
      physical["data_type"] = data_type #+2.5
      # physical["mac_power"] = 0.026 #12nm
      # physical["mac_time"] = 1 #12nm
      physical["mac_area"] = 0.000783 * 1.5 #mm^2 , 45nm + an overhead for wiring
      communication_power_orig, data_rate_orig = calc_communication(orig_channels, 0, physical, comm, enable_qam=False)
      comm["max_data_rate"] = data_rate_orig / 10**6
      print("MAX DATA RATE:", comm["max_data_rate"])

      #for the model we can increase the network time
      physical["network_time"] = 1 * 10**6 / 2 #network_time * 4 #2KHz
      computation_power = 0

      arch = dnn_arch.copy()

      original_channels = arch["input_channels"]
      ratio = (channels / original_channels)
      arch["input_channels"] = channels
      physical["num_channels"] = channels

      scale_network(arch, ratio, network_input_dependency)

      _, _, layers_res, macs = \
        run_paper_network_wrapper(arch, physical, comm, soc, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

      # #Check what's the power costs of the non sensing part without communication
      # non_sensing_power_no_comm = 0 #(power_per_channel_comm * orig_channels - communication_power_orig * 10**3) / orig_channels * arch["last_output"]

      non_sensing_power_no_comm = calc_non_sensing_power_no_comm(physical, comm, soc, arch["last_output"])

      communication_power, _ = calc_communication(arch["last_output"], 0, physical, comm, enable_qam=False)

      non_sensing_power_comm = None #non_sensing_power_no_comm + communication_power * 10**3

      #layers_res = layers_res - non_sensing_power_comm

      # #Update to include also the reset of the non sensing power
      # layers_res = layers_res + non_sensing_power_no_comm

      non_sensing_power_consumption_layers_plot.append(layers_res)

      computation_area = physical["mac_area"] * macs

      #power consumption
      non_sensing_power_consumption_layers = layers_res



      #power consumption
      power_consumption_channels = sensing_power_consumption + non_sensing_power_consumption_layers
      # if channels == orig_channels + step: #corner case to remove the first computation and fix plot
      #   power_consumption_plot.append(None)
      # else:
      #   power_consumption_plot.append(power_consumption_channels)

      #this is where I should end the while loop

      # power_consumption_plot.append(power_consumption_channels)

      total_power_budget = power_budget_per_area * (sensing_area + non_sensing_area + computation_area)
      non_sensing_power_budget = total_power_budget - sensing_power_consumption

      #power budget
      print("Main: power budget is:", total_power_budget, "non-sensing budget is:", non_sensing_power_budget)

      if show==True or power_consumption_channels < total_power_budget*scale_power:
        total_power_budget_plot.append(total_power_budget) #
        sensing_power_consumption_plot.append(sensing_power_consumption) #
        non_sensing_power_consumption_comm_plot.append(non_sensing_power_comm)
        power_consumption_plot.append(power_consumption_channels)


    #Prepare x axis - channel number
    x_val = channels
    if show==True or power_consumption_channels < total_power_budget*scale_power:
      x_axis.append(x_val)

    if show==False and power_consumption_channels > total_power_budget*scale_power:
      # ctr = ctr + 1
      # if ctr == 256:
      break


  dnn_type = dnn_arch["DNN"]

  #intersection
  y1 = np.array(power_consumption_plot)
  y2 = np.array(total_power_budget_plot)
  x = x_axis

  x_intersections = list()
  y_intersections = list()

  x_intersections, y_intersections = intersection_finder(y1, y2, x)

  if show == True:
    fig, ax = plt.subplots(figsize=(12, 6))

    mid_index = len(total_power_budget_plot) // 2
    if dnn_type == "DN-CNN":
      label_x, label_y = x_axis[mid_index-120], total_power_budget_plot[mid_index-70]
    else:
      label_x, label_y = x_axis[mid_index-120], total_power_budget_plot[mid_index-90]

    if dnn_type == "DN-CNN":
      angle = np.arctan2(total_power_budget_plot[mid_index+1] - total_power_budget_plot[mid_index-1], x_axis[mid_index+1] - x_axis[mid_index-1]) * 18 #/ np.pi
    else:
      angle = np.arctan2(total_power_budget_plot[mid_index+1] - total_power_budget_plot[mid_index-1], x_axis[mid_index+1] - x_axis[mid_index-1]) * 30 #/ np.pi

    print(angle)
    angle_degrees = np.degrees(angle)
    print(angle_degrees)

    ax.axvline(x=1024, color='b', linestyle=':', linewidth=2, label="Data rate limit")
    label_xx, label_yy = 1024-40, power_consumption_plot[mid_index * 2 - 40]
    print(total_power_budget_plot[-1])

    plt.plot(x_axis, total_power_budget_plot, linestyle='--', linewidth=4, color='red', label=None)
    plt.text(label_x, label_y, 'Power Budget', rotation=angle_degrees-360, ha='center', va='center', color='red', weight='bold', fontsize=18)
    #plt.text(label_xx, label_yy, 'Zeng et al.', rotation=90, ha='center', va='center', color='blue', fontsize=12)
    #plt.text(x_axis[int(max_channels/2 / step)] + 0.1, total_power_budget_plot[int(max_channels/2 / step)], 'Power Budget', color='red')


    #plt.plot(x_axis, non_sensing_power_consumption_pipe_plot, label='Non-Sensing Power Consumption (DN-CNN)', color='purple')

    if dnn_type == "DN-CNN":
      ax.axvline(x=1408 ,ymin=0, ymax=0.4, color='gray', linestyle='--', linewidth=2, label=None)
    else:
      ax.axvline(x=2232 ,ymin=0, ymax=0.8, color='gray', linestyle='--', linewidth=2, label=None)

    plt.plot(x_axis, sensing_power_consumption_plot, label='Sensing Power Consumption', color='blue', linewidth=3)

    if dnn_type == "DN-CNN":
      plt.plot(x_axis, non_sensing_power_consumption_layers_plot, label=f"Non-Sensing Power ({dnn_type})", color='purple', linewidth=3)
    else:
      plt.plot(x_axis, non_sensing_power_consumption_layers_plot, label=f"Non-Sensing Power ({dnn_type})", color='purple', linewidth=3)

    plt.plot(x_axis, non_sensing_power_consumption_comm_plot, label='Non-Sensing Power (Comm.)', color=(0.8, 0.6, 0), linewidth=3)
    #plt.plot(x_axis, non_sensing_power_consumption_no_pipe_plot, label='Non-Sensing Power Consumption No Pipe')
    plt.plot(x_axis, power_consumption_plot, label='Total Power Consumption', color='green', linewidth=3)

    plt.xticks(x_axis[::tick_ratio], fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Number of Active Channels', fontsize=22)
    plt.ylabel('Power [mW]', fontsize=22)
    plt.grid(axis='y')
    #plt.title('Number of Active Channels vs. Power')
    plt.legend(fontsize=15,loc='upper left', bbox_to_anchor=(0, 1), ncol=1)

    plt.tight_layout()

    #plt.savefig('comp_layers1.pdf')
    plt.savefig(f"figures/{name}_{dnn_type}_layers_scale.pdf")

    plt.show()

  return x_intersections, y_intersections, x_axis, sensing_power_consumption_plot, non_sensing_power_consumption_layers_plot, power_consumption_plot, total_power_budget_plot

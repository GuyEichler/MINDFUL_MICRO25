from settings.structs import *
from framework.helper import *
import math
import sys
from framework.QAM import *


def check_on_chip_potential(layer_operations, layer_sequences, output_per_layer, physical, comm, pipelined):

  mac_time = physical["mac_time"]
  network_time = physical["network_time"]
  #physical_mac_num = round(physical_constraints["power_budget"] / physical_constraints["mac_power"])
  # communication_power_no_compute, data_rate_no_compute = calc_communication_params_no_specific(physical["num_channels"], 0, physical, comm)
  # power_constraint = min(physical["power_budget"], communication_power_no_compute*10**3)
  power_constraint = physical["power_budget"]
  mac_power = physical["mac_power"]
  physical_mac_num = round(power_constraint / mac_power)

  print(power_constraint, mac_power, physical_mac_num)

  #If network can't fit because MAC power is too high
  minimum_time = 0
  if physical_mac_num == 0:
    for i in range(len(layer_operations)):
      del layer_operations[-1]
      del layer_sequences[-1]
      del output_per_layer[-1]
    return

  for i in range(len(layer_operations)):
    if pipelined == False:
      minimum_time = minimum_time + layer_time(layer_operations[i], layer_sequences[i], physical_mac_num, mac_time)
    else:
      minimum_time = max(minimum_time, layer_time(layer_operations[i], layer_sequences[i], physical_mac_num, mac_time))

  #minimum_time = sum(layer_sequences) * mac_time
  while minimum_time >= network_time: #full network doesn't fit
    print("REMOVING LAYER: MINIMUM TIME: ", minimum_time, "NETWORK TIME: ", network_time)
    #remove last layer
    del layer_operations[-1]
    del layer_sequences[-1]
    del output_per_layer[-1]
    #minimum_time = sum(layer_sequences) * mac_time
    minimum_time = 0
    for i in range(len(layer_operations)):
      if pipelined == False:
        minimum_time = minimum_time + layer_time(layer_operations[i], layer_sequences[i], physical_mac_num, mac_time)
      else:
        minimum_time = max(minimum_time, layer_time(layer_operations[i], layer_sequences[i], physical_mac_num, mac_time))

  return layer_operations, layer_sequences, output_per_layer


def calc_communication(output_size, total_compute_time, physical, comm, enable_qam=False):

  #Calculate what is left for communication
  data_type = physical["data_type"]
  data_size_to_transmit = output_size * data_type
  energy_per_bit = comm["energy_per_bit"] #50 * 10**-12 J - this can be taken from any communication IP

  #energy_recursion, power_recursion, total_time_recursion = print_network_statistics(layer_operations, time_per_layer, mac_power, macs_per_layer, True)
  network_time = physical["network_time"] #ns
  time_left = network_time - total_compute_time #ns
  #print("time left:", time_left, "ns")
  data_rate = data_size_to_transmit / (time_left * 10**-9) #bit/sec
  max_data_rate = comm["max_data_rate"]

  print("max data rate", max_data_rate)
  print("data rate", data_rate/10**6)
  #print(data_size_to_transmit)
  #print(time_left * 10**-9)

  if enable_qam == True and data_rate/10**6 > max_data_rate:
    antenna_utilization = math.ceil(data_rate/10**6 / max_data_rate)
    print("Antenna utilization:", antenna_utilization)
    energy_per_bit = QAM(n=antenna_utilization, efficiency=comm["qam_efficiency"], ber=comm["qam_ber"], path_loss=comm["qam_path_loss"], margin=comm["qam_margin"])

  communication_power = energy_per_bit * data_rate #J / bit * bit / sec = J / sec = W

  print("Communication power:", round(communication_power * 10**3, 4), "mW data rate:", round(data_rate / 10**6, 4), "Mbit/sec")
  # print(data_size_to_transmit)
  # print(time_left)

  return communication_power, data_rate


def pipelined_layer_optimizer(min_macs, max_macs, num_layers, time_per_layer, layer_operations, layer_sequences, mac_time, network_time, mac_power, max_layer, macs_per_layer):

  for i in range(min_macs, max_macs+1): #assuming this loop always breaks at some point due to previous checks
    if sum(max_layer) < num_layers: #finish finding a minimum for each layer
      for j in range(num_layers):
        if max_layer[j] == 1:
          continue
        else:
          time_per_layer[j] = layer_time(layer_operations[j], layer_sequences[j], i, mac_time)
          if time_per_layer[j] < network_time:
            print("Found minimum MACs for layer", j+1, "MAC number is", i)
            max_layer[j] = 1
            macs_per_layer[j] = i
    else:
      print("Minimum MACs found for all layers pipelined")
      break

  pipeline_stages = num_layers #initialize
  time_per_stage = time_per_layer.copy()
  macs_per_stage = macs_per_layer.copy()
  previous_pipeline_stages = pipeline_stages
  merged_layers = [[] for _ in range(num_layers)]

  for i in range(num_layers):
    merged_layers[i].append(i)

  #Try to merge pipeline stages for reusability of MACs
  while pipeline_stages > 1:
    # print("Stages:", pipeline_stages)
    # print("time per stage:", time_per_stage)
    minimum_couple = time_per_stage[0] + time_per_stage[1] # try to merge the first two stages first
    index = 0
    found = False
    #check if stages can be merged
    for i in range(pipeline_stages):
      if i < (pipeline_stages - 1):
        # print("Trying to merge", i, "and", i+1)
        new_couple = time_per_stage[i] + time_per_stage[i+1]
        if new_couple <= network_time and new_couple < minimum_couple:
          minimum_couple = new_couple
          index = i
          found = True
      else: #last iteration - check minimum couple
        if found == True:
          # print("Merging", index, "and", index+1)
          time_per_stage[index] = time_per_stage[index] + time_per_stage[index+1]
          macs_per_stage[index] = max(macs_per_stage[index], macs_per_stage[index+1])
          merged_layers[index].extend(merged_layers[index+1]) #merge the stages
          del time_per_stage[index+1]
          del macs_per_stage[index+1]
          del merged_layers[index+1]


    previous_pipeline_stages = pipeline_stages
    pipeline_stages = len(time_per_stage)
    if pipeline_stages == previous_pipeline_stages: #Number of stages didn't change
      print("Final stages:", pipeline_stages)
      print("Final time per stage:", time_per_stage)
      break


  return max_layer, macs_per_stage, time_per_stage, merged_layers

def layer_optimizer(min_macs, max_macs, num_layers, time_per_layer, layer_operations, layer_sequences, mac_time, network_time, mac_power, max_layer, macs_per_layer):

  #new maximum MACs
  new_max = 0
  max_index = 0
  max_energy_addition = 0
  total_time_per_mac_num = 0

  for i in range(min_macs, max_macs+1):
    for j in range(num_layers):
      if max_layer[j] == 1:
        continue
      else:
        time_per_layer[j] = layer_time(layer_operations[j], layer_sequences[j], i, mac_time)

    total_time_per_mac_num = sum(time_per_layer)

    #print(total_time_per_mac_num, i, time_per_layer)

    if total_time_per_mac_num < network_time:
      #print("Minimum MACs found:", i, "total time:", total_time_per_mac_num, "ns")
      for j in range(num_layers):
        if max_layer[j] == 0:
          print("Time for layer", j+1, "is:", time_per_layer[j], "ns")

      #print("Computation power:", i*mac_power, "mW")
      #return
      new_max = i
      break

  #check which layer was previously the bottleneck

  tentative_time = 0;

  for i in range(num_layers):
    if new_max == 1 and max_layer[i] == 0:
      #1 MAC is sufficient for the rest of the network
      max_layer[i] = 1
      macs_per_layer[i] = new_max
      max_index = i
    elif max_layer[i] == 0:
      print("Checking layer", i+1) #check what happens if we remove a MAC
      tentative_time = layer_time(layer_operations[i], layer_sequences[i], new_max-1, mac_time)

      if (tentative_time - time_per_layer[i] + total_time_per_mac_num) > network_time:
        #1st heuristic - find the first layer that pushes the time over the limit
        print("Layer", i+1, "1st heuristic")
        max_index = i
        break
      elif ((tentative_time - time_per_layer[i]) * mac_power * (new_max-1)) > max_energy_addition:
        #2nd heuristic - check if the layer adds the most energy
        #Checking if this layer adds the most energy consumption if MACs are reduced
        #print("Least energy efficient layer:", i+1)
        print("Layer", i+1, "2nd heuristic")
        max_energy_addition = (tentative_time - time_per_layer[i]) * mac_power * (new_max-1)
        max_index = i

  #max_index = time_per_layer_local.index(max(time_per_layer_local));
  max_layer[max_index] = 1
  macs_per_layer[max_index] = new_max
  print("Most demanding layer:", max_index+1)

  print("Minimum MACs found:", new_max)
  print("Network total time:", total_time_per_mac_num, "ns")
  print("Computation power:", new_max*mac_power, "mW")

  return new_max



def pipelined_network_optimizer(layer_operations, layer_sequences, physical):

  #power_constraint = min(physical["power_budget"], communication_power_no_compute)
  power_constraint = physical["power_budget"]
  physical_mac_num = round(power_constraint / physical["mac_power"])
  mac_time = physical["mac_time"]
  mac_power = physical["mac_power"]
  network_time = physical["network_time"]
  min_macs = physical["min_macs"]

  max_macs = min(sum(layer_operations), int(physical_mac_num))
  #no_recursion_max_macs = max(layer_operations)
  num_layers = len(layer_operations)
  time_per_layer = list()
  #no_recursion_time_per_layer = list()
  max_layer = list()
  macs_per_layer = list()

  if len(layer_operations) != len(layer_sequences):
    sys.exit("Bad configuration!")

  #initialize list
  for i in range(num_layers):
    time_per_layer.append(0)
    max_layer.append(0)
    macs_per_layer.append(0)

  new_max = 0

  max_layer, macs_per_stage, time_per_stage, merged_layers = \
    pipelined_layer_optimizer(min_macs, max_macs, num_layers, time_per_layer, layer_operations, layer_sequences, mac_time, network_time, mac_power, max_layer, macs_per_layer)


  return time_per_layer, macs_per_layer, macs_per_stage, time_per_stage, merged_layers





def network_optimizer(layer_operations, layer_sequences, physical):

  #power_constraint = min(physical["power_budget"], communication_power_no_compute)
  power_constraint = physical["power_budget"]
  physical_mac_num = round(power_constraint / physical["mac_power"])
  mac_time = physical["mac_time"]
  mac_power = physical["mac_power"]
  network_time = physical["network_time"]
  min_macs = physical["min_macs"]

  max_macs = min(max(layer_operations), int(physical_mac_num))
  no_recursion_max_macs = max(layer_operations)
  num_layers = len(layer_operations)
  time_per_layer = list()
  no_recursion_time_per_layer = list()
  max_layer = list()
  macs_per_layer = list()

  #initialize list
  for i in range(num_layers):
    time_per_layer.append(0)
    max_layer.append(0)
    macs_per_layer.append(0)

  new_max = 0
  max_optimizations = num_layers

  #first optimization - find the number of MACs that can run al layers
  new_max = \
    layer_optimizer(min_macs, max_macs, num_layers, time_per_layer, layer_operations, layer_sequences, mac_time, network_time, mac_power, max_layer, macs_per_layer)

  max_macs = new_max
  no_recursion_time_per_layer = time_per_layer.copy()
  no_recursion_max_macs = new_max

  #optimize according to the number of layers
  for i in range(max_optimizations-1):
    print("\nOptimization", i+1)

    new_max = \
      layer_optimizer(min_macs, max_macs, num_layers, time_per_layer, layer_operations, layer_sequences, mac_time, network_time, mac_power, max_layer, macs_per_layer)

    max_macs = new_max

    if sum(max_layer) == num_layers:
      #optimized all layers
      break

  return time_per_layer, macs_per_layer, no_recursion_time_per_layer, no_recursion_max_macs


def print_pipelined_network_statistics(layer_operations, layer_sequences, time_per_layer, time_per_stage, physical, macs_per_layer, macs_per_stage, merged_layers):

  full_energy = 0
  full_power = 0
  merge_energy = 0
  merge_power = 0
  num_layers = len(layer_operations)
  mac_power = physical["mac_power"]
  mac_time = physical["mac_time"]

  print("\nFinal results per layer fully pipelined:")
  for i in range(num_layers):
    print("Layer", i+1, "time:", time_per_layer[i], "MACs:", macs_per_layer[i])
    #energy is fixed per layer, all layers run at once, sum energy to calculate the power
    #full_energy = time_per_layer[i] * 10**-9 * mac_power * 10**-3 * macs_per_layer[i] #J
    layer_energy = layer_operations[i] * layer_sequences[i] * mac_time * 10**-9 * mac_power * 10**-3 #J
    full_energy = full_energy + layer_energy
    #full_power = full_power + mac_power * 10**-3 * macs_per_layer[i] #W
    # divide each layer's energy by the time it runs to calculate the added power
    full_power = full_power + layer_energy / (time_per_layer[i] * 10**-9) #W

  print("Maximum time for network layer:", max(time_per_layer), "ns")
  print("Total energy for network computation:", round(full_energy, 14), "J")

  # average_power = energy / (max(time_per_layer) * 10**-9) #W

  print("Total network power when fully pipelined:", round(full_power * 10**3, 4), "mW")

  print("\nFinal results per layer allowing to merge stages:")
  #print("With power gating:")
  for i in range(len(macs_per_stage)):
    print("Stage", i+1, "time:", time_per_stage[i], "MACs:", macs_per_stage[i], "Layers:", merged_layers[i])
    merge_energy = 0
    for j in merged_layers[i]:
      #merge_energy = merge_energy + time_per_layer[j] * 10**-9 * mac_power * 10**-3 * macs_per_layer[i] #J
      merge_energy = merge_energy + layer_operations[j] * layer_sequences[j] * mac_time * 10**-9 * mac_power * 10**-3 #J
    merge_power = merge_power + merge_energy / (time_per_stage[i] * 10**-9) #W - sum average power for each stage to get total power

  print("Maximum time for network stage:", max(time_per_stage), "ns")
  print("Total energy for network computation:", round(merge_energy, 14), "J")
  print("Total network power when merge pipelined:", round(merge_power * 10**3, 4), "mW")

  return full_energy, full_power, max(time_per_layer), merge_energy, merge_power, max(time_per_stage)


def pipelined_network_statistics(layer_operations, time_per_layer, time_per_stage, mac_power, macs_per_layer, macs_per_stage, merged_layers):

  full_energy = 0
  full_power = 0
  merge_energy = 0
  merge_power = 0

  #print("\nFinal results per layer fully pipelined:")
  for i in range(len(layer_operations)):
    #print("Layer", i+1, "time:", time_per_layer[i], "MACs:", macs_per_layer[i])
    full_energy = time_per_layer[i] * 10**-9 * mac_power * 10**-3 * macs_per_layer[i] #J
    full_power = full_power + mac_power * 10**-3 * macs_per_layer[i] #W

  #print("Maximum time for network layer:", max(time_per_layer), "ns")
  #print("Total energy for network computation:", round(full_energy, 10), "J")

  # average_power = energy / (max(time_per_layer) * 10**-9) #W

  #print("Total network power when fully pipelined:", round(full_power * 10**3, 4), "mW")

  #print("\nFinal results per layer allowing to merge stages:")
  #print("With power gating:")
  for i in range(len(macs_per_stage)):
    #print("Stage", i+1, "time:", time_per_stage[i], "MACs:", macs_per_stage[i], "Layers:", merged_layers[i])
    merge_energy = 0
    for j in merged_layers[i]:
      merge_energy = merge_energy + time_per_layer[j] * 10**-9 * mac_power * 10**-3 * macs_per_layer[i] #J
    merge_power = merge_power + merge_energy / (time_per_stage[i] * 10**-9) #W - sum average power for each stage to get total power

  #print("Maximum time for network stage:", max(time_per_stage), "ns")
  #print("Total energy for network computation:", round(merge_energy, 10), "J")
  #print("Total network power when merge pipelined:", round(merge_power * 10**3, 4), "mW")

  return full_energy, full_power, max(time_per_layer), merge_energy, merge_power, max(time_per_stage)


def print_network_statistics(layer_operations, layer_sequences, time_per_layer, physical, macs_per_layer):

  energy = 0
  mac_power = physical["mac_power"]
  mac_time = physical["mac_time"]

  print("\nFinal time per layer:")
  for i in range(len(layer_operations)):
    print("Layer", i+1, "time:", time_per_layer[i])
    energy = energy + layer_operations[i] * layer_sequences[i] * mac_time * 10**-9 * mac_power * 10**-3 #J
    # if recursive == True:
    #   energy = energy + time_per_layer[i] * 10**-9 * mac_power * 10**-3 * macs_per_layer[i] #J
    # else:
    #   energy = energy + time_per_layer[i] * 10**-9 * mac_power * 10**-3 * macs_per_layer #J

  print("MACs per layer:", macs_per_layer)
  print("Total computation time for network:", sum(time_per_layer), "ns")
  print("Total computation energy for network:", round(energy, 12), "J")

  average_power = energy / (sum(time_per_layer) * 10**-9) #W

  print("Average network power:", round(average_power * 10**3, 4), "mW")

  return energy, average_power, sum(time_per_layer)

def network_statistics(layer_operations, time_per_layer, mac_power, macs_per_layer, recursive):

  energy = 0

  #print("\nFinal time per layer:")
  for i in range(len(layer_operations)):
    #print("Layer", i+1, "time:", time_per_layer[i])
    if recursive == True:
      energy = energy + time_per_layer[i] * 10**-9 * mac_power * 10**-3 * macs_per_layer[i] #J
    else:
      energy = energy + time_per_layer[i] * 10**-9 * mac_power * 10**-3 * macs_per_layer #J

  #print("Total time for network:", sum(time_per_layer), "ns")
  #print("Total energy for network computation:", round(energy, 11), "J")

  average_power = energy / (sum(time_per_layer) * 10**-9) #W

  #print("Average network power:", round(average_power * 10**3, 4), "mW")

  return energy, average_power, sum(time_per_layer)



def optimize_full_network_parameters(layer_operations, layer_sequences, mac_power, output_per_layer, physical, comm):

  communication_power = 0
  data_rate = 0
  compute_power = 0
  compute_time = 0
  output_size = 0

  no_pipeline = no_pipeline_dict.copy()

  original_network_length = len(layer_operations)
  no_pipeline_layer_operations = layer_operations.copy()
  no_pipeline_layer_sequences = layer_sequences.copy()
  no_pipeline_output_per_layer = output_per_layer.copy()

  #first calculate no pipeline design with/without power gating

  output_size = no_pipeline_output_per_layer[-1] #get the size of the last on-chip layer
  recursion_time_per_layer, recursion_macs_per_layer, no_recursion_time_per_layer, no_recursion_max_macs = \
    network_optimizer(no_pipeline_layer_operations, no_pipeline_layer_sequences, physical)

  no_pipeline["time_per_layer"] = recursion_time_per_layer
  no_pipeline["macs_per_layer"] = recursion_macs_per_layer
  no_pipeline["no_recursion_time_per_layer"] = no_recursion_time_per_layer
  no_pipeline["no_recursion_max_macs"] = no_recursion_max_macs

  #No power gating data
  energy_no_recursion, compute_power, compute_time = \
    print_network_statistics(no_pipeline_layer_operations, no_pipeline_layer_sequences, no_recursion_time_per_layer, physical, no_recursion_max_macs)

  communication_power, data_rate = calc_communication(output_size, compute_time, physical, comm)

  no_pipeline["no_recursion_compute_energy"] = energy_no_recursion
  no_pipeline["no_recursion_compute_power"] = compute_power
  no_pipeline["no_recursion_compute_time"] = compute_time
  no_pipeline["no_recursion_communication_power"] = communication_power
  no_pipeline["no_recursion_communication_data_rate"] = data_rate

  #Power gating recursion
  energy_recursion, compute_power, compute_time = \
    print_network_statistics(no_pipeline_layer_operations, no_pipeline_layer_sequences, recursion_time_per_layer, physical, recursion_macs_per_layer)

  communication_power, data_rate = calc_communication(output_size, compute_time, physical, comm)

  no_pipeline["recursion_compute_energy"] = energy_recursion
  no_pipeline["recursion_compute_power"] = compute_power
  no_pipeline["recursion_compute_time"] = compute_time
  no_pipeline["recursion_communication_power"] = communication_power
  no_pipeline["recursion_communication_data_rate"] = data_rate
  no_pipeline["output_size"] = output_size

  return no_pipeline


def optimize_pipelined_network_parameters(layer_operations, layer_sequences, mac_power, output_per_layer, physical, comm):

  communication_power = 0
  data_rate = 0
  compute_power = 0
  compute_time = 0
  output_size = 0

  pipeline = pipeline_dict.copy()

  original_network_length = len(layer_operations)
  pipeline_layer_operations = layer_operations.copy()
  pipeline_layer_sequences = layer_sequences.copy()
  pipeline_output_per_layer = output_per_layer.copy()

  #Calculate pipelined design with full pipeline (every layer) or by allowing to merge layers (stage)

  output_size = pipeline_output_per_layer[-1]
  pipeline_time_per_layer, pipeline_macs_per_layer, macs_per_stage, time_per_stage, merged_layers = \
    pipelined_network_optimizer(pipeline_layer_operations, pipeline_layer_sequences, physical)

  pipeline["time_per_layer"] = pipeline_time_per_layer
  pipeline["macs_per_layer"] = pipeline_macs_per_layer
  pipeline["macs_per_stage"] = macs_per_stage
  pipeline["time_per_stage"] = time_per_stage
  pipeline["merged_layers"] = merged_layers

  full_energy, full_power, max_time_per_layer, merge_energy, merge_power, max_time_per_stage = \
    print_pipelined_network_statistics(pipeline_layer_operations, pipeline_layer_sequences, pipeline_time_per_layer, time_per_stage, physical, pipeline_macs_per_layer, macs_per_stage, merged_layers)

  pipeline["full_compute_energy"] = full_energy
  pipeline["full_compute_power"] = full_power
  pipeline["max_time_per_layer"] = max_time_per_layer
  pipeline["merge_compute_energy"] = merge_energy
  pipeline["merge_compute_power"] = merge_power
  pipeline["max_time_per_stage"] = max_time_per_stage

  communication_power, data_rate = calc_communication(output_size, 0, physical, comm)

  pipeline["communication_power"] = communication_power
  pipeline["communication_data_rate"] = data_rate
  pipeline["output_size"] = output_size

  return pipeline


def summary(communication_power, data_rate, compute_power, compute_time):

  print("\nFinal design parameters:")
  if data_rate / 10**6 > communication["max_data_rate"]:
    print("Data rate",round(data_rate/10**6),"Mb/s is higher than the maximum!")

  communication_time = physical_constraints["network_time"] - compute_time #ns

  total_energy = compute_power * compute_time/10**9 + communication_power * communication_time/10**9 #J
  average_power = total_energy / (physical_constraints["network_time"]/10**9)

  print("Total energy computation+communication:", round(total_energy, 11), "J")
  print("Average power computation+communication:", round(average_power*10**3, 4), "mW")
  print("Communication data rate:", round(data_rate/10**6, 4), "Mbit/sec")


def results_summary(results, physical, comm):

  type = results.type
  communication_power = results.communication_power
  data_rate = results.communication_data_rate
  compute_power = results.computation_power
  compute_time = results.computation_time

  print("\nDesign type:", type)

  print("Final design parameters:")
  if data_rate / 10**6 > comm["max_data_rate"]:
    print("Data rate",round(data_rate/10**6),"Mb/s is higher than the maximum!")

  #communication_time = physical["network_time"] - compute_time #ns
  communication_time = physical["network_time"]

  total_energy = compute_power * compute_time/10**9 + communication_power * communication_time/10**9 #J
  average_power = total_energy / (physical["network_time"]/10**9)

  print("Total energy computation+communication:", round(total_energy, 11), "J")
  print("Average power computation+communication:", round(average_power*10**3, 4), "mW")
  print("Communication data rate:", round(data_rate/10**6, 4), "Mbit/sec")
  print("Computation time:", round(compute_time), "ns")

  return average_power*10**3


def calc_non_sensing_power_no_comm(physical, comm, soc, output_size):

  power_consumption = soc["power_consumption"]
  sensing_area = soc["sensing_area"]
  total_area = soc["total_area"]
  non_sensing_area = total_area - sensing_area
  max_comm_channels = soc["max_comm_channels"]
  orig_channels = soc["active_channels"]

  communication_power_orig, data_rate_orig = calc_communication(orig_channels, 0, physical, comm, enable_qam=False)

  power_per_channel_comm = power_consumption * non_sensing_area/total_area / max_comm_channels

  non_sensing_power_no_comm = (power_per_channel_comm * orig_channels - communication_power_orig * 10**3) / orig_channels * output_size

  return non_sensing_power_no_comm

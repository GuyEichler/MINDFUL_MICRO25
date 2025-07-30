import math
import sys
from settings.structs import *
from framework.optimization import *
import numpy as np

def scale_network(arch, ratio, network_input_dependency=1):

  channels = arch["input_channels"]

  if arch["DNN"] == "MLP":

    arch["input_size"] = channels * arch["timestamps"]

    ratio = 1 + (ratio - 1) * network_input_dependency

    update_hidden = [arch["hidden_size"][0]] * len(arch["hidden_size"] * math.ceil(ratio)) #.copy()
    for i in range(len(update_hidden)):
      N = len(update_hidden)
      update_hidden[i] = round(update_hidden[i] * (1 + ((N - i - 1) / (N - 1)) * (ratio - 1)))
      print(update_hidden)

    arch["hidden_size"] = update_hidden

  elif arch["DNN"] == "DN-CNN":

    arch["growth_factor"] = math.ceil(ratio * network_input_dependency * arch["growth_factor"])

  else: #S2S

    arch["input_size"] = channels

    ratio = 1 + (ratio - 1) * network_input_dependency

    encoder_layers = arch["encoder_layers"]
    decoder_layers = arch["decoder_layers"]
    arch["encoder_layers"] = encoder_layers * math.ceil(ratio)
    arch["decoder_layers"] = decoder_layers * math.ceil(ratio)


def calc_densenet(input_size, input_channels, filter_size, output_channels, padding, striding, pooling, growth_factor, reduce_factor, inner_dense_layers, dense_blocks):

  layer_operations = list()
  layer_sequences = list()
  output_per_layer = list()

  #input conv layer
  current_input_size = input_size
  current_input_channels = input_channels
  current_filter_size = filter_size[0]
  current_output_channels = 2 * growth_factor
  current_padding = padding[0]
  current_striding = striding[0]
  current_output_size = math.floor((current_input_size-current_filter_size+2*current_padding) / current_striding + 1)

  total_mac_operations = current_input_channels * current_filter_size * current_output_channels * current_output_size
  layer_operations.append(current_output_size * current_output_channels)
  layer_sequences.append(current_input_channels * current_filter_size)
  output_per_layer.append(current_output_channels * current_output_size)

  #dense blocks and transition layers
  total_layers_num = 1 + dense_blocks + (dense_blocks - 1)
  dense = True
  #growth_factor
  #inner_layers
  #reduce_factor

  #need number of dense blocks, inner dense blocks, transition blocks, final block

  for i in range(1,total_layers_num):
    if(dense == True):
      dense_input_channels = current_output_channels
      for j in range(inner_dense_layers):
        current_input_size = current_output_size
        current_input_channels = current_output_channels #dense_input_channels + j * growth_factor
        current_filter_size = filter_size[i]
        current_padding = padding[i]
        current_striding = striding[i]
        current_pooling = pooling[i]
        current_output_channels = growth_factor
        current_output_size = math.floor((current_input_size-current_filter_size+2*current_padding) / current_striding + 1)

        total_mac_operations = current_input_channels * current_filter_size * current_output_channels * current_output_size
        layer_operations.append(current_output_size * current_output_channels)
        layer_sequences.append(current_input_channels * current_filter_size)
        current_output_size = math.floor((current_output_size-current_pooling) / current_pooling + 1) #after pooling
        current_output_channels = dense_input_channels + (j+1) * growth_factor
        output_per_layer.append(current_output_channels * current_output_size)

      #current_output_channels = current_output_channels + current_input_channels
      dense = False #switch to transition layer
    else: #transition
      current_input_size = current_output_size
      current_input_channels = current_output_channels
      current_filter_size = filter_size[i]
      current_padding = padding[i]
      current_striding = striding[i]
      current_pooling = pooling[i]
      current_output_channels = math.floor(reduce_factor * current_input_channels)
      current_output_size = math.floor((current_input_size-current_filter_size+2*current_padding) / current_striding + 1) #before pooling

      total_mac_operations = current_input_channels * current_filter_size * current_output_channels * current_output_size
      layer_operations.append(current_output_size * current_output_channels)
      layer_sequences.append(current_input_channels * current_filter_size)
      current_output_size = math.floor((current_output_size-current_pooling) / current_pooling + 1) #after pooling
      output_per_layer.append(current_output_channels * current_output_size)
      dense = True #switch to dense

  #output layer
  current_input_size = current_output_size
  current_input_channels = current_output_channels
  current_filter_size = filter_size[total_layers_num]
  current_padding = padding[total_layers_num]
  current_striding = striding[total_layers_num]
  current_pooling = pooling[total_layers_num]
  current_output_channels = output_channels
  current_output_size = math.floor((current_input_size-current_filter_size+2*current_padding) / current_striding + 1) #before pooling
  total_mac_operations = current_input_channels * current_filter_size * current_output_channels * current_output_size
  layer_operations.append(current_output_size * current_output_channels)
  layer_sequences.append(current_input_channels * current_filter_size)
  current_output_size = math.floor((current_output_size-current_pooling) / current_pooling + 1) #after pooling
  output_per_layer.append(current_output_channels * current_output_size)

  return layer_operations, layer_sequences, output_per_layer

def calc_mlp(input_size, hidden_size, output_size):

  # input_size = 2048
  # hidden_size = [256, 256]
  # output_size = 40
  #num_hidden_layers = 2

  layer_operations = list()
  layer_sequences = list()
  output_per_layer = list()

  #input layer
  num_operations = hidden_size[0]
  sequence_length = input_size
  layer_operations.append(num_operations)
  layer_sequences.append(sequence_length)
  output_per_layer.append(num_operations)

  #hidden layers
  for i in range(len(hidden_size)):

    num_operations = hidden_size[i]
    if i == 0:
      sequence_length = hidden_size[0]
    else:
      sequence_length = hidden_size[i-1]
    layer_operations.append(num_operations)
    layer_sequences.append(sequence_length)
    output_per_layer.append(num_operations)

  #output layer
  num_operations = output_size
  sequence_length = hidden_size[-1]
  layer_operations.append(num_operations)
  layer_sequences.append(sequence_length)
  output_per_layer.append(num_operations)

  print(layer_operations, layer_sequences, output_per_layer)

  return layer_operations, layer_sequences, output_per_layer


def calc_s2s(input_size, encoder_gru_layers, encoder_directions, decoder_gru_layers, decoder_directions, output_size):

  layer_operations = list()
  layer_sequences = list()
  output_per_layer = list()

  hidden_size = output_size

  #Encoder GRU gates operations - encoder block
  reset_operations = hidden_size + hidden_size
  reset_sequence = max(input_size, hidden_size) #simplify take the maximum
  update_operations = hidden_size + hidden_size
  update_sequence = max(input_size, hidden_size) #simplify take the maximum
  new_operations = hidden_size + hidden_size
  new_sequence = max(input_size, hidden_size) #simplify take the maximum
  hidden_operations = 2
  hidden_sequence = hidden_size

  num_operations = reset_operations+update_operations
  sequence_length = max(reset_sequence, update_sequence) + new_sequence + hidden_sequence

  if encoder_directions > 1:
    num_operations = num_operations * encoder_directions

  #Cannot pipeline inside encoder layer - only sequence grows with internal layers
  for i in range(encoder_gru_layers-1):
    sequence_length = sequence_length + hidden_size * 3

  #Store encoder
  layer_operations.append(num_operations)
  layer_sequences.append(sequence_length)
  output_per_layer.append(hidden_size * encoder_directions)

  #Attention block - 4 matrix multiplications - 2 1 1
  attention_input = hidden_size
  num_operations = 2 * hidden_size #arch specific
  sequence_length = hidden_size * 3 #arch specific
  layer_operations.append(num_operations)
  layer_sequences.append(sequence_length)
  output_per_layer.append(hidden_size)

  #Decoder GRU block
  reset_operations = hidden_size + hidden_size
  reset_sequence = hidden_size * 2
  update_operations = hidden_size + hidden_size
  update_sequence = hidden_size * 2
  new_operations = hidden_size + hidden_size
  new_sequence = hidden_size * 2
  hidden_operations = 2
  hidden_sequence = hidden_size

  num_operations = reset_operations+update_operations
  sequence_length = max(reset_sequence, update_sequence) + new_sequence + hidden_sequence

  if decoder_directions > 1:
    num_operations = num_operations * decoder_directions

  for i in range(decoder_gru_layers-1):
    sequence_length = sequence_length + hidden_size * 3

  #Store decoder
  layer_operations.append(num_operations)
  layer_sequences.append(sequence_length)
  output_per_layer.append(hidden_size * decoder_directions)


  print(layer_operations, layer_sequences, output_per_layer)

  return layer_operations, layer_sequences, output_per_layer

def calc_dnn(dnn_arch):

  arch = dnn_arch.copy()

  if arch["DNN"] == "MLP":

    input_size = arch["input_size"]
    hidden_size = arch["hidden_size"]
    output_size = arch["output_size"]
    layer_operations, layer_sequences, output_per_layer = calc_mlp(input_size, hidden_size, output_size)
    return layer_operations, layer_sequences, output_per_layer

  elif arch["DNN"] == "DN-CNN":

    input_size = arch["input_size"]
    input_channels = arch["input_channels"]
    filter_size = arch["filter_size"]
    output_channels = arch["output_channels"]
    padding = arch["padding"]
    striding = arch["striding"]
    pooling = arch["pooling"]
    inner_dense_layers = arch["inner_dense_layers"]
    dense_blocks = arch["dense_blocks"]
    growth_factor = arch["growth_factor"]
    reduce_factor = arch["reduce_factor"]
    layer_operations, layer_sequences, output_per_layer = \
      calc_densenet(input_size, input_channels, filter_size, output_channels, padding, striding, pooling, growth_factor, reduce_factor, inner_dense_layers, dense_blocks)
    return layer_operations, layer_sequences, output_per_layer

  else: #S2S

    input_size = arch["input_size"]
    encoder_gru_layers = arch["encoder_layers"]
    encoder_directions = arch["encoder_directions"]
    decoder_gru_layers = arch["decoder_layers"]
    decoder_directions = arch["decoder_directions"]
    output_size = arch["output_size"]
    layer_operations, layer_sequences, output_per_layer = calc_s2s(input_size, encoder_gru_layers, encoder_directions, decoder_gru_layers, decoder_directions, output_size)
    return layer_operations, layer_sequences, output_per_layer

def mlp(physical, comm, arch, allow_reduction=False, manual_num_layers=None):

  #for each number of MAC units possible from 1 to maximum parallelism
  #calculate the time for each layer
  #take the first number of MACs that fit

  #physical constraints for the network
  mac_time = physical["mac_time"]
  mac_power = physical["mac_power"]
  network_time = physical["network_time"]
  min_macs = physical["min_macs"] #1
  num_channels = physical["num_channels"]

  #network architecture parameters
  input_size = arch["input_size"] #2048
  hidden_size = arch["hidden_size"] #[256, 256]
  output_size = arch["output_size"] #40

  #Compute the total operations by layer in the network
  layer_operations, layer_sequences, output_per_layer = calc_mlp(input_size, hidden_size, output_size)
  total_accumulations = total_macs(layer_operations, layer_sequences)
  original_network_length = len(layer_operations)

  #Store results
  no_compute = no_compute_dict.copy()

  print("\nCommunication power without any on-chip computation:")
  communication_power_no_compute, data_rate_no_compute = \
    calc_communication(num_channels, 0, physical, comm, enable_qam=True)

  no_compute["communication_power"] = communication_power_no_compute
  no_compute["data_rate"] = data_rate_no_compute

  #  if adjust_budget == True:
  if manual_num_layers != None:
    #keep budget for compute under the original communication budget
    #physical["power_budget"] = communication_power_no_compute*10**3
    while len(layer_operations) != manual_num_layers:
      del layer_operations[-1]
      del layer_sequences[-1]
      del output_per_layer[-1]

  #Check if potentially the network can fit on chip - to find a lower maximum for MACs
  check_on_chip_potential(layer_operations, layer_sequences, output_per_layer, physical, comm, pipelined=False)
  print("Number of potential layers on-chip:", len(layer_operations), "out of", original_network_length,"\n")

  if len(layer_operations) < original_network_length:
    print("Full network does not fit! Channels", physical["num_channels"]) #skip network optimizer and go to communication
    # if allow_reduction == False and manual_num_layers == None:
    #   sys.exit("Closing program")
    # else:
    print("Reducing network")

  if len(layer_operations) == 0:
    print("Network does not fit! Channels", physical["num_channels"]) #skip network optimizer and go to communication
    # if manual_num_layers == None:
    #   sys.exit("Closing program")
    # else:
    #send back empty computation
    no_pipeline = no_compute
    return no_compute, no_pipeline

  no_pipeline = \
    optimize_full_network_parameters(layer_operations, layer_sequences, mac_power, output_per_layer, physical, comm)

  return no_compute, no_pipeline


def mlp_pipe(physical, comm, arch, allow_reduction=False, manual_num_layers=None):

  #for each number of MAC units possible from 1 to maximum parallelism
  #calculate the time for each layer
  #take the first number of MACs that fit

  #physical constraints for the network
  mac_time = physical["mac_time"]
  mac_power = physical["mac_power"]
  network_time = physical["network_time"]
  min_macs = physical["min_macs"] #1
  num_channels = physical["num_channels"]

  #network architecture parameters
  input_size = arch["input_size"] #2048
  hidden_size = arch["hidden_size"] #[256, 256]
  output_size = arch["output_size"] #40

  #Compute the total operations by layer in the network
  layer_operations, layer_sequences, output_per_layer = calc_mlp(input_size, hidden_size, output_size)
  total_accumulations = total_macs(layer_operations, layer_sequences)
  original_network_length = len(layer_operations)

  #Store results
  no_compute = no_compute_dict.copy()

  print("\nCommunication power without any on-chip computation:")
  communication_power_no_compute, data_rate_no_compute = \
    calc_communication(num_channels, 0, physical, comm, enable_qam=True)

  no_compute["communication_power"] = communication_power_no_compute
  no_compute["data_rate"] = data_rate_no_compute

  #if adjust_budget == True:
  if manual_num_layers != None:
    #keep budget for compute under the original communication budget
    #physical["power_budget"] = communication_power_no_compute*10**3
    while len(layer_operations) != manual_num_layers:
      del layer_operations[-1]
      del layer_sequences[-1]
      del output_per_layer[-1]

  #Check if potentially the network can fit on chip - to find a lower maximum for MACs
  check_on_chip_potential(layer_operations, layer_sequences, output_per_layer, physical, comm, pipelined=True)
  print("Number of potential layers on-chip:", len(layer_operations), "out of", original_network_length,"\n")

  if len(layer_operations) < original_network_length:
    print("Full network does not fit! Channels", physical["num_channels"]) #skip network optimizer and go to communication
    # if allow_reduction == False and manual_num_layers == None:
    #   sys.exit("Closing program")
    # else:
    print("Reducing network")

  if len(layer_operations) == 0:
    print("Network does not fit! Channels", physical["num_channels"]) #skip network optimizer and go to communication
    # if manual_num_layers == None:
    #   sys.exit("Closing program")
    # else:
    #send back empty computation
    pipeline = no_compute
    return no_compute, pipeline

  pipeline = \
    optimize_pipelined_network_parameters(layer_operations, layer_sequences, mac_power, output_per_layer, physical, comm)

  return no_compute, pipeline


def densenet(physical, comm, arch, allow_reduction=False, manual_num_layers=None):

  #for each number of MAC units possible from 1 to maximum parallelism
  #calculate the time for each layer
  #take the first number of MACs that fit

  #physical constraints for the network
  mac_time = physical["mac_time"]
  mac_power = physical["mac_power"]
  network_time = physical["network_time"] #0.5*10**6 ns
  min_macs = physical["min_macs"] #1
  num_channels = physical["num_channels"]

  #network architecture parameters
  input_size = arch["input_size"]
  input_channels = arch["input_channels"]
  filter_size = arch["filter_size"]
  output_channels = arch["output_channels"]
  padding = arch["padding"]
  striding = arch["striding"]
  pooling = arch["pooling"]
  inner_dense_layers = arch["inner_dense_layers"]
  dense_blocks = arch["dense_blocks"]
  growth_factor = arch["growth_factor"]
  reduce_factor = arch["reduce_factor"]

  layer_operations, layer_sequences, output_per_layer = \
    calc_densenet(input_size, input_channels, filter_size, output_channels, padding, striding, pooling, growth_factor, reduce_factor, inner_dense_layers, dense_blocks)

  total_accumulations = total_macs(layer_operations, layer_sequences)
  original_network_length = len(layer_operations)

  #Store results
  no_compute = no_compute_dict.copy()

  print("\nCommunication power without any on-chip computation:")
  communication_power_no_compute, data_rate_no_compute = \
    calc_communication(num_channels, 0, physical, comm, enable_qam=True)

  no_compute["communication_power"] = communication_power_no_compute
  no_compute["data_rate"] = data_rate_no_compute

  if manual_num_layers != None:
    #keep budget for compute under the original communication budget
    #physical["power_budget"] = communication_power_no_compute*10**3
    while len(layer_operations) != manual_num_layers:
      del layer_operations[-1]
      del layer_sequences[-1]
      del output_per_layer[-1]

  #Check if potentially the network can fit on chip - to find a lower maximum for MACs
  check_on_chip_potential(layer_operations, layer_sequences, output_per_layer, physical, comm, pipelined=False)
  print("Number of potential layers on-chip:", len(layer_operations), "out of", original_network_length,"\n")

  #print("\nmanual num layers in networks no pipeline:",manual_num_layers,"\n")

  if len(layer_operations) < original_network_length:
    print("Full network does not fit! Channels", physical["num_channels"]) #skip network optimizer and go to communication
    # if allow_reduction == False and manual_num_layers == None:
    #   sys.exit("Closing program")
    # else:
    print("Reducing network")

  if len(layer_operations) == 0:
    print("Network does not fit! Channels", physical["num_channels"]) #skip network optimizer and go to communication
    # if manual_num_layers == None:
    #   sys.exit("Closing program")
    # else:
    #send back empty computation
    no_pipeline = no_compute
    return no_compute, no_pipeline

  no_pipeline = \
    optimize_full_network_parameters(layer_operations, layer_sequences, mac_power, output_per_layer, physical, comm)

  return no_compute, no_pipeline


def densenet_pipe(physical, comm, arch, allow_reduction=False, manual_num_layers=None):

  #for each number of MAC units possible from 1 to maximum parallelism
  #calculate the time for each layer
  #take the first number of MACs that fit

  #physical constraints for the network
  mac_time = physical["mac_time"]
  mac_power = physical["mac_power"]
  network_time = physical["network_time"] #0.5*10**6 ns
  min_macs = physical["min_macs"] #1
  num_channels = physical["num_channels"]

  #network architecture parameters
  input_size = arch["input_size"]
  input_channels = arch["input_channels"]
  filter_size = arch["filter_size"]
  output_channels = arch["output_channels"]
  padding = arch["padding"]
  striding = arch["striding"]
  pooling = arch["pooling"]
  inner_dense_layers = arch["inner_dense_layers"]
  dense_blocks = arch["dense_blocks"]
  growth_factor = arch["growth_factor"]
  reduce_factor = arch["reduce_factor"]

  layer_operations, layer_sequences, output_per_layer = \
    calc_densenet(input_size, input_channels, filter_size, output_channels, padding, striding, pooling, growth_factor, reduce_factor, inner_dense_layers, dense_blocks)

  total_accumulations = total_macs(layer_operations, layer_sequences)
  original_network_length = len(layer_operations)

  #Store results
  no_compute = no_compute_dict.copy()

  print("\nCommunication power without any on-chip computation:")
  communication_power_no_compute, data_rate_no_compute = \
    calc_communication(num_channels, 0, physical, comm, enable_qam=True)

  no_compute["communication_power"] = communication_power_no_compute
  no_compute["data_rate"] = data_rate_no_compute

  if manual_num_layers != None:
    #keep budget for compute under the original communication budget
    #physical["power_budget"] = communication_power_no_compute*10**3
    while len(layer_operations) != manual_num_layers:
      del layer_operations[-1]
      del layer_sequences[-1]
      del output_per_layer[-1]

  #Check if potentially the network can fit on chip - to find a lower maximum for MACs
  check_on_chip_potential(layer_operations, layer_sequences, output_per_layer, physical, comm, pipelined=True)
  print("Number of potential layers on-chip:", len(layer_operations), "out of", original_network_length,"\n")

  if len(layer_operations) < original_network_length:
    print("Full network does not fit! Channels", physical["num_channels"]) #skip network optimizer and go to communication
    # if allow_reduction == False and manual_num_layers == None:
    #   sys.exit("Closing program")
    # else:
    print("Reducing network")

  if len(layer_operations) == 0:
    print("Network does not fit! Channels", physical["num_channels"]) #skip network optimizer and go to communication
    # if manual_num_layers == None:
    #   sys.exit("Closing program")
    # else:
    #send back empty computation
    pipeline = no_compute
    return no_compute, pipeline

  pipeline = \
    optimize_pipelined_network_parameters(layer_operations, layer_sequences, mac_power, output_per_layer, physical, comm)

  return no_compute, pipeline



def s2s(physical, comm, arch, allow_reduction=False, manual_num_layers=None):

  #for each number of MAC units possible from 1 to maximum parallelism
  #calculate the time for each layer
  #take the first number of MACs that fit

  #physical constraints for the network
  mac_time = physical["mac_time"]
  mac_power = physical["mac_power"]
  network_time = physical["network_time"]
  min_macs = physical["min_macs"] #1
  num_channels = physical["num_channels"]

  #network architecture parameters
  input_size = arch["input_size"] #128
  encoder_gru_layers = arch["encoder_layers"]
  encoder_directions = arch["encoder_directions"]
  decoder_gru_layers = arch["decoder_layers"]
  decoder_directions = arch["decoder_directions"]
  output_size = arch["output_size"] #40

  #Compute the total operations by layer in the network
  layer_operations, layer_sequences, output_per_layer = calc_s2s(input_size, encoder_gru_layers, encoder_directions, decoder_gru_layers, decoder_directions, output_size)
  total_accumulations = total_macs(layer_operations, layer_sequences)
  original_network_length = len(layer_operations)

  #Store results
  no_compute = no_compute_dict.copy()

  print("\nCommunication power without any on-chip computation:")
  communication_power_no_compute, data_rate_no_compute = \
    calc_communication(num_channels, 0, physical, comm, enable_qam=True)

  no_compute["communication_power"] = communication_power_no_compute
  no_compute["data_rate"] = data_rate_no_compute

  #  if adjust_budget == True:
  if manual_num_layers != None:
    #keep budget for compute under the original communication budget
    #physical["power_budget"] = communication_power_no_compute*10**3
    while len(layer_operations) != manual_num_layers:
      del layer_operations[-1]
      del layer_sequences[-1]
      del output_per_layer[-1]

  #Check if potentially the network can fit on chip - to find a lower maximum for MACs
  check_on_chip_potential(layer_operations, layer_sequences, output_per_layer, physical, comm, pipelined=False)
  print("Number of potential layers on-chip:", len(layer_operations), "out of", original_network_length,"\n")

  if len(layer_operations) < original_network_length:
    print("Full network does not fit! Channels", physical["num_channels"]) #skip network optimizer and go to communication
    # if allow_reduction == False and manual_num_layers == None:
    #   sys.exit("Closing program")
    # else:
    print("Reducing network")

  if len(layer_operations) == 0:
    print("Network does not fit! Channels", physical["num_channels"]) #skip network optimizer and go to communication
    # if manual_num_layers == None:
    #   sys.exit("Closing program")
    # else:
    #send back empty computation
    no_pipeline = no_compute
    return no_compute, no_pipeline

  no_pipeline = \
    optimize_full_network_parameters(layer_operations, layer_sequences, mac_power, output_per_layer, physical, comm)

  return no_compute, no_pipeline


def paper_s2s_pipe(physical, comm, arch, allow_reduction=False, manual_num_layers=None):

  #for each number of MAC units possible from 1 to maximum parallelism
  #calculate the time for each layer
  #take the first number of MACs that fit

  #physical constraints for the network
  mac_time = physical["mac_time"]
  mac_power = physical["mac_power"]
  network_time = physical["network_time"]
  min_macs = physical["min_macs"] #1
  num_channels = physical["num_channels"]

  #network architecture parameters
  input_size = arch["input_size"] #128
  encoder_gru_layers = arch["encoder_layers"]
  encoder_directions = arch["encoder_directions"]
  decoder_gru_layers = arch["decoder_layers"]
  decoder_directions = arch["decoder_directions"]
  output_size = arch["output_size"] #40

  #Compute the total operations by layer in the network
  layer_operations, layer_sequences, output_per_layer = \
    calc_s2s(input_size, encoder_gru_layers, encoder_directions, decoder_gru_layers, decoder_directions, output_size)
  total_accumulations = total_macs(layer_operations, layer_sequences)
  original_network_length = len(layer_operations)

  total_accumulations = total_macs(layer_operations, layer_sequences)
  original_network_length = len(layer_operations)
  physical["power_budget"] = max(layer_operations) * physical["mac_power"]

  #Store results
  no_compute = no_compute_dict.copy()

  print("\nCommunication power without any on-chip computation:")
  communication_power_no_compute, data_rate_no_compute = \
    calc_communication(num_channels, 0, physical, comm, enable_qam=True)

  no_compute["communication_power"] = communication_power_no_compute
  no_compute["data_rate"] = data_rate_no_compute

  #if adjust_budget == True:
  if manual_num_layers != None:
    #keep budget for compute under the original communication budget
    #physical["power_budget"] = communication_power_no_compute*10**3
    while len(layer_operations) != manual_num_layers:
      del layer_operations[-1]
      del layer_sequences[-1]
      del output_per_layer[-1]

    original_network_length = len(layer_operations)

  #Check if potentially the network can fit on chip - to find a lower maximum for MACs
  check_on_chip_potential(layer_operations, layer_sequences, output_per_layer, physical, comm, pipelined=True)
  print("Number of potential layers on-chip:", len(layer_operations), "out of", original_network_length,"\n")

  if len(layer_operations) < original_network_length:
    print("Full network does not fit! Channels", physical["num_channels"]) #skip network optimizer and go to communication
    # if allow_reduction == False and manual_num_layers == None:
    #   sys.exit("Closing program")
    # else:
    if allow_reduction == False:
      pipeline = no_compute
      print("Can't reduce network")
      return no_compute, pipeline
    else:
      print("Reducing network")

  if len(layer_operations) == 0:
    print("Network does not fit! Channels", physical["num_channels"]) #skip network optimizer and go to communication
    # if manual_num_layers == None:
    #   sys.exit("Closing program")
    # else:
    #send back empty computation
    pipeline = no_compute
    return no_compute, pipeline

  pipeline = \
    optimize_pipelined_network_parameters(layer_operations, layer_sequences, mac_power, output_per_layer, physical, comm)

  physical["mac_num"] = sum(pipeline["macs_per_layer"])

  return no_compute, pipeline


def paper_densenet_pipe(physical, comm, arch, allow_reduction=False, manual_num_layers=None):

  #for each number of MAC units possible from 1 to maximum parallelism
  #calculate the time for each layer
  #take the first number of MACs that fit

  #physical constraints for the network
  mac_time = physical["mac_time"]
  mac_power = physical["mac_power"]
  network_time = physical["network_time"] #0.5*10**6 ns
  min_macs = physical["min_macs"] #1
  num_channels = physical["num_channels"]

  #network architecture parameters
  input_size = arch["input_size"]
  input_channels = arch["input_channels"]
  filter_size = arch["filter_size"]
  output_channels = arch["output_channels"]
  padding = arch["padding"]
  striding = arch["striding"]
  pooling = arch["pooling"]
  inner_dense_layers = arch["inner_dense_layers"]
  dense_blocks = arch["dense_blocks"]
  growth_factor = arch["growth_factor"]
  reduce_factor = arch["reduce_factor"]

  layer_operations, layer_sequences, output_per_layer = \
    calc_densenet(input_size, input_channels, filter_size, output_channels, padding, striding, pooling, growth_factor, reduce_factor, inner_dense_layers, dense_blocks)

  total_accumulations = total_macs(layer_operations, layer_sequences)
  original_network_length = len(layer_operations)
  physical["power_budget"] = max(layer_operations) * physical["mac_power"]

  #Store results
  no_compute = no_compute_dict.copy()

  print("\nCommunication power without any on-chip computation:")
  communication_power_no_compute, data_rate_no_compute = \
    calc_communication(num_channels, 0, physical, comm, enable_qam=True)

  no_compute["communication_power"] = communication_power_no_compute
  no_compute["data_rate"] = data_rate_no_compute

  if manual_num_layers != None:
    #keep budget for compute under the original communication budget
    #physical["power_budget"] = communication_power_no_compute*10**3
    while len(layer_operations) > manual_num_layers:
      del layer_operations[-1]
      del layer_sequences[-1]
      del output_per_layer[-1]

    original_network_length = len(layer_operations)

  #Check if potentially the network can fit on chip - to find a lower maximum for MACs
  check_on_chip_potential(layer_operations, layer_sequences, output_per_layer, physical, comm, pipelined=True)
  print("Number of potential layers on-chip:", len(layer_operations), "out of", original_network_length,"\n")

  if len(layer_operations) < original_network_length:
    print("Full network does not fit! Channels", physical["num_channels"]) #skip network optimizer and go to communication
    # if allow_reduction == False and manual_num_layers == None:
    #   sys.exit("Closing program")
    # else:
    if allow_reduction == False:
      pipeline = no_compute
      print("Can't reduce network")
      return no_compute, pipeline
    else:
      print("Reducing network")


  if len(layer_operations) == 0:
    print("Network does not fit! Channels", physical["num_channels"]) #skip network optimizer and go to communication
    # if manual_num_layers == None:
    #   sys.exit("Closing program")
    # else:
    #send back empty computation
    pipeline = no_compute
    return no_compute, pipeline

  pipeline = \
    optimize_pipelined_network_parameters(layer_operations, layer_sequences, mac_power, output_per_layer, physical, comm)

  physical["mac_num"] = sum(pipeline["macs_per_layer"])

  return no_compute, pipeline



def paper_mlp_pipe(physical, comm, arch, allow_reduction=False, manual_num_layers=None):

  #for each number of MAC units possible from 1 to maximum parallelism
  #calculate the time for each layer
  #take the first number of MACs that fit

  #physical constraints for the network
  mac_time = physical["mac_time"]
  mac_power = physical["mac_power"]
  network_time = physical["network_time"]
  min_macs = physical["min_macs"] #1
  num_channels = physical["num_channels"]

  #network architecture parameters
  input_size = arch["input_size"] #2048
  hidden_size = arch["hidden_size"] #[256, 256]
  output_size = arch["output_size"] #40

  layer_operations, layer_sequences, output_per_layer = calc_mlp(input_size, hidden_size, output_size)

  total_accumulations = total_macs(layer_operations, layer_sequences)
  original_network_length = len(layer_operations)
  physical["power_budget"] = max(layer_operations) * physical["mac_power"]

  #Store results
  no_compute = no_compute_dict.copy()

  print("\nCommunication power without any on-chip computation:")
  communication_power_no_compute, data_rate_no_compute = \
    calc_communication(num_channels, 0, physical, comm, enable_qam=True)

  no_compute["communication_power"] = communication_power_no_compute
  no_compute["data_rate"] = data_rate_no_compute

  if manual_num_layers != None:
    #keep budget for compute under the original communication budget
    #physical["power_budget"] = communication_power_no_compute*10**3
    while len(layer_operations) > manual_num_layers:
      del layer_operations[-1]
      del layer_sequences[-1]
      del output_per_layer[-1]

    original_network_length = len(layer_operations)

  #Check if potentially the network can fit on chip - to find a lower maximum for MACs
  check_on_chip_potential(layer_operations, layer_sequences, output_per_layer, physical, comm, pipelined=True)
  print("Number of potential layers on-chip:", len(layer_operations), "out of", original_network_length,"\n")

  if len(layer_operations) < original_network_length:
    print("Full network does not fit! Channels", physical["num_channels"]) #skip network optimizer and go to communication
    # if allow_reduction == False and manual_num_layers == None:
    #   sys.exit("Closing program")
    # else:
    if allow_reduction == False:
      pipeline = no_compute
      print("Can't reduce network")
      return no_compute, pipeline
    else:
      print("Reducing network")


  if len(layer_operations) == 0:
    print("Network does not fit! Channels", physical["num_channels"]) #skip network optimizer and go to communication
    # if manual_num_layers == None:
    #   sys.exit("Closing program")
    # else:
    #send back empty computation
    pipeline = no_compute
    return no_compute, pipeline

  pipeline = \
    optimize_pipelined_network_parameters(layer_operations, layer_sequences, mac_power, output_per_layer, physical, comm)

  physical["mac_num"] = sum(pipeline["macs_per_layer"])

  return no_compute, pipeline


##### RUN FUNCTIONS #####


def run_paper_network(network, arch, physical, comm, soc, total_power_budget=np.inf, allow_reduction=False, aggressive=False, budget=False, check_data_rate=False):

  no_recursive_res = results("not recursive")
  recursive_res = results("recursive")
  pipeline_res = results("pipeline")
  merge_pipeline_res = results("merged pipeline")
  no_compute_res = results("no computation")

  merge_macs = 0

  no_compute, pipeline = network(physical, comm, arch, allow_reduction)

  #print(no_compute)
  no_compute_res.communication_power = no_compute["communication_power"]
  no_compute_res.communication_data_rate = no_compute["data_rate"]
  no_compute_res.computation_power = 0
  no_compute_res.computation_time = 0
  no_comp_power = results_summary(no_compute_res, physical, comm)

  if no_compute != pipeline: #should always enter
    pipeline_res.communication_power = pipeline["communication_power"] # + \
      # calc_non_sensing_power_no_comm(physical, comm, soc, arch["last_output"])/10**3
    pipeline_res.communication_data_rate = pipeline["communication_data_rate"]
    pipeline_res.computation_power = pipeline["full_compute_power"]
    pipeline_res.computation_time = pipeline["max_time_per_layer"]
    full_pipe_power = results_summary(pipeline_res, physical, comm)

    merge_pipeline_res.communication_power = pipeline["communication_power"] # + \
      # calc_non_sensing_power_no_comm(physical, comm, soc, arch["last_output"])/10**3
    merge_pipeline_res.communication_data_rate = pipeline["communication_data_rate"]
    merge_pipeline_res.computation_power = pipeline["merge_compute_power"]
    merge_pipeline_res.computation_time = pipeline["max_time_per_stage"]
    merge_pipeline_res.macs_per_stage = pipeline["macs_per_stage"]
    merge_macs = sum(pipeline["macs_per_stage"])
    merge_pipe_power = results_summary(merge_pipeline_res, physical, comm)
  else: #should never enter this
    full_pipe_power = None #no_comp_power
    merge_pipe_power = None #no_comp_power

  bound = no_comp_power if budget==False else total_power_budget #physical["power_budget"]

  #print("\nBOUND IS:", bound, "\n")

  #no_compute_tmp = no_compute
  if aggressive == True:

    minimum_full = 0
    minimum_merge = 0
    new_num_layers = 0

    if no_compute != pipeline: #should always enter
      new_num_layers = len(pipeline["time_per_layer"])
      minimum_full = new_num_layers
      minimum_merge = new_num_layers
      min_full_power = full_pipe_power
      min_merge_power = merge_pipe_power
    else:
      min_full_power = no_comp_power
      min_merge_power = no_comp_power

    while new_num_layers > 0: # and min_merge_power > bound:
      new_num_layers = new_num_layers - 1
      print("\nAGGRESSIVE OPTIMIZATION LAYERS:", new_num_layers, "MINIMUM POWER:", min_merge_power, "BUDGET:", bound)
      no_compute_tmp, pipeline_check = network(physical, comm, arch, allow_reduction, new_num_layers)

      if no_compute_tmp != pipeline_check: #should always enter
        if (check_data_rate == True and pipeline_check["communication_data_rate"]/10**6 < comm["max_data_rate"]) or check_data_rate == False:
          last_output =  pipeline_check["output_size"]
          pipeline_res.communication_power = pipeline_check["communication_power"] # + \
            # calc_non_sensing_power_no_comm(physical, comm, soc, last_output)/10**3
          pipeline_res.communication_data_rate = pipeline_check["communication_data_rate"]
          pipeline_res.computation_power = pipeline_check["full_compute_power"]
          pipeline_res.computation_time = pipeline_check["max_time_per_layer"]
          full_power_check = results_summary(pipeline_res, physical, comm)
          if full_power_check < min_full_power:
            min_full_power = full_power_check
            minimum_full = new_num_layers

          merge_pipeline_res.communication_power = pipeline_check["communication_power"] # + \
            # calc_non_sensing_power_no_comm(physical, comm, soc, last_output)/10**3
          merge_pipeline_res.communication_data_rate = pipeline_check["communication_data_rate"]
          merge_pipeline_res.computation_power = pipeline_check["merge_compute_power"]
          merge_pipeline_res.computation_time = pipeline_check["max_time_per_stage"]
          merge_pipeline_res.macs_per_stage = pipeline_check["macs_per_stage"]
          merge_power_check = results_summary(merge_pipeline_res, physical, comm)
          if merge_power_check < min_merge_power:
            print("\nAGGRESSIVE OPTIMIZATION NEW MINIMUM LAYERs:", new_num_layers)
            min_merge_power = merge_power_check
            minimum_merge = new_num_layers
            arch["reduced_layers"] = new_num_layers
            arch["last_output"] =  last_output
            merge_macs = sum(pipeline_check["macs_per_stage"])
        else:
          print("REDUCED NETWORK DOES NOT MEET NEEDS:", pipeline_check["communication_data_rate"]/10**6, comm["max_data_rate"], "LAYERS:", new_num_layers)

    #Store minimums
    #if min_full_power < bound and min_full_power != no_comp_power:
    if min_full_power != no_comp_power:
      #print("\nAGGRESSIVE OPTIMIZATION: FULL POWER LOWER THAN BOUND")
      full_pipe_power = min_full_power
    else:
      full_pipe_power = None #no_comp_power
      #print("\nAGGRESSIVE OPTIMIZATION: FULL POWER HIGHER THAN BOUND", full_pipe_power)
    #if min_merge_power < bound and min_merge_power != no_comp_power:
    if min_merge_power != no_comp_power:
      print("\nAGGRESSIVE OPTIMIZATION: MINIMUM LAYERS:", arch["reduced_layers"])
      print("AGGRESSIVE OPTIMIZATION POWER:", min_merge_power)
      merge_pipe_power = min_merge_power
    else:
      merge_pipe_power = None #no_comp_power
      #print("\nAGGRESSIVE OPTIMIZATION: MERGE POWER HIGHER THAN BOUND")

    print("\nMinimum full:", minimum_full, full_pipe_power ,"Minimum merge:", minimum_merge, merge_pipe_power, "\n")

  return no_comp_power, full_pipe_power, merge_pipe_power, merge_macs

def run_paper_network_wrapper(arch, physical, comm, soc, total_power_budget=np.inf, allow_reduction=False, aggressive=False, budget=False, check_data_rate=False):

  no_comp_power = 0
  full_pipe_power = 0
  merge_pipe_power = 0
  merge_macs = 0

  if arch["DNN"] == "MLP":

    no_comp_power, full_pipe_power, merge_pipe_power, merge_macs = \
      run_paper_network(paper_mlp_pipe, arch, physical, comm, soc, allow_reduction=allow_reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

  elif arch["DNN"] == "DN-CNN":

    no_comp_power, full_pipe_power, merge_pipe_power, merge_macs = \
      run_paper_network(paper_densenet_pipe, arch, physical, comm, soc, allow_reduction=allow_reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

  else: #S2S

    no_comp_power, full_pipe_power, merge_pipe_power, merge_macs = \
      run_paper_network(paper_s2s_pipe, arch, physical, comm, soc, allow_reduction=allow_reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

  return no_comp_power, full_pipe_power, merge_pipe_power, merge_macs

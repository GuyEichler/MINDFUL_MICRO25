class results:
  def __init__(self, type):
    self.type = type
    self.communication_power = 0
    self.communication_data_rate = 0
    self.computation_power = 0
    self.computation_time = 0
    self.macs_per_stage = []

  def __str__(self):
    return f"type: {self.type}\n" + \
      f"communication power: {self.communication_power}\n" + \
      f"communication data rate: {self.communication_data_rate}\n" + \
      f"computation power: {self.computation_power}\n" + \
      f"computation time: {self.computation_time}"

no_compute_dict = {
  "communication_power" : 0,
  "data_rate" : 0
}

no_pipeline_dict = {
  "time_per_layer" : [],
  "macs_per_layer" : [],
  "no_recursion_time_per_layer" : 0,
  "no_recursion_max_macs" : 1,
  "recursion_compute_energy" : 0,
  "recursion_compute_power" : 0,
  "recursion_compute_time" : 0,
  "recursion_communication_power" : 0,
  "recursion_communication_data_rate" : 0,
  "no_recursion_compute_energy" : 0,
  "no_recursion_compute_power" : 0,
  "no_recursion_compute_time" : 0,
  "no_recursion_communication_power" : 0,
  "no_recursion_communication_data_rate" : 0,
  "output_size" : 0
}

pipeline_dict = {
  "time_per_layer" : [],
  "macs_per_layer" : [],
  "macs_per_stage" : [],
  "time_per_stage" : [],
  "merged_layers" : [],
  "full_compute_energy" : 0,
  "full_compute_power" : 0,
  "max_time_per_layer" : 0,
  "merge_compute_energy" : 0,
  "merge_compute_power" : 0,
  "max_time_per_stage" : 0,
  "communication_power" : 0,
  "communication_data_rate" : 0,
  "output_size" : 0
}

communication = {
  "energy_per_bit" : 50 * 10**-12, #J
  "max_data_rate" : 500000, #Mb/s
  "qam_efficiency" : 15, #%
  "qam_ber" : 1*10**-6, #bit error rate
  "qam_path_loss" : 6, #60dB
  "qam_margin" : 2 #20dB
  }

physical_constraints = {
  "mac_time" : 2, #ns - 45nm
  "mac_power" : 0.05, #mW - 45nm (0.0005mW BitFusion)
  "network_time" : 0.5*10**6, #0.5 #ns - according to sample rate from channels 2KHz
  "min_macs" : 1,
  "data_type" : 8, #bit - set later by the SoC
  "power_budget" : 100, #mW
  "num_channels" : 128, #128 - changes according to scaling
  "mac_area" : 0,
  "mac_num" : 0
  }

s2s_architecture = {
  "DNN" : "S2S",
  "input_channels": 128,
  "input_size" : 128, #electrodes
  "encoder_layers" : 2,
  "encoder_directions" : 2,
  "decoder_layers" : 2,
  "decoder_directions" : 2,
  "output_size" : 40
  }

mlp_architecture = {
  "DNN" : "MLP",
  "input_channels": 128,
  "input_size" : 128*16, #electrodes * timestamps
  "timestamps" : 16,
  "hidden_size" : [256, 256],
  "output_size" : 40,
  "num_layers" : 4,
  "reduced_layers" : 4,
  "last_output" : 40
  }

densenet_architecture = {
  "DNN" : "DN-CNN",
  "input_size" : 36, #timestamps
  "input_channels" : 128, #electrodes
  "filter_size" : [3, 3, 1, 3, 1, 3, 1],
  "output_channels" : 40, #final output
  "padding" : [1, 1, 0, 1, 0, 1, 0],
  "striding" : [1, 1, 1, 1, 1, 1, 1],
  "pooling" : [1, 1, 3, 1, 3, 1, 3],
  "growth_factor" : 30,
  "reduce_factor": 0.6,
  "dense_blocks" : 3,
  "inner_dense_layers" : 2,
  "reduced_layers" : 10,
  "last_output" : 40
  }

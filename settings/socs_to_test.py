from settings.soc_struct import *
from settings.structs import *
import math as mt

########################################

#Default parameters for 45nm technology after synthesis
physical_specs45 = physical_constraints.copy()
physical_specs45["mac_time"] = 2 #ns
physical_specs45["mac_power"] = 0.05 #mW
physical_specs45["mac_area"] = 0.000783 * 1.5 #mm^2
physical_specs45["mac_num"] = 0 #To be determined by optimization
physical_specs45["min_macs"] = 1
physical_specs45["network_time"] = 0 #To be determined by SoC
physical_specs45["power_budget"] = 100000000 #mW - Set to high - upper limit only
physical_specs45["num_channels"] = 128 #To be determined by SoC

#Plot colors
colors = [
  "red",
  "blue",
  "green",
  "orange",
  "purple",
  "cyan",
  "magenta",
  "brown"
]

########################################

#Data structures that hold the parameters of the SoCs to test.
#The parameters reflect the ones of the SoCs scaled to 1024 channels from Section 4.1 in the MINDFUL paper.

socs = list()

bisc = list()

bisc_soc = soc_parameters.copy()
bisc_soc["Name"] = "1"
bisc_soc["active_channels"] = 1024
bisc_soc["max_channels"] = 65536
bisc_soc["sensing_area"] = 6.8 * 7.4 #mm^2
bisc_soc["total_area"] = 144 #mm^2
bisc_soc["power_consumption"] = 38.8 #mW
bisc_soc["max_comm_channels"] = 1024
bisc_soc["data_type"] = 10
bisc_soc["sampling_period"] = 0.5*10**6 / 4 #8KHz
bisc_soc["power_density_budget"] = 0.4 #mw/mm^2
bisc_soc["budget_cutoff"] = [None, None] #to be determined

bisc_comm_specs = communication.copy()
bisc_comm_specs["energy_per_bit"] = 50 * 10**-12 #J
bisc_comm_specs["max_data_rate"] = 50000 #J - a placeholder upper limit
bisc_comm_specs["qam_efficiency"] = 15 #% - nominal
bisc_comm_specs["qam_ber"] = 1*10**-6 # nominal
bisc_comm_specs["qam_path_loss"] = 6 #60dB nominal
bisc_comm_specs["qam_margin"] = 2 #20dB nominal

bisc.append(bisc_soc)
bisc.append(bisc_comm_specs)
socs.append(bisc)

ibis = list()

scale = 49152/1024

ibis_soc = soc_parameters.copy()
ibis_soc["Name"] = "2"
ibis_soc["active_channels"] = 49152/scale
ibis_soc["max_channels"] = 49152/scale
ibis_soc["sensing_area"] = 5.76 * 7.68 #mm^2
ibis_soc["total_area"] = 144 #mm^2
ibis_soc["power_consumption"] = 48 #mW
ibis_soc["max_comm_channels"] = 49152/scale
ibis_soc["data_type"] = 10
ibis_soc["sampling_period"] = 1 / 176 * 10**9 / scale#176 / 1 sec
ibis_soc["power_density_budget"] = 0.4 #mw/mm^2
ibis_soc["budget_cutoff"] = [None, None] #to be determined

ibis_comm_specs = communication.copy()
ibis_comm_specs["energy_per_bit"] = 50 * 10**-12 #J
ibis_comm_specs["max_data_rate"] = 50000 #J - a placeholder upper limit
ibis_comm_specs["qam_efficiency"] = 15 #% - nominal
ibis_comm_specs["qam_ber"] = 1*10**-6 # nominal
ibis_comm_specs["qam_path_loss"] = 6 #60dB nominal
ibis_comm_specs["qam_margin"] = 2 #20dB nominal

ibis.append(ibis_soc)
ibis.append(ibis_comm_specs)
socs.append(ibis)

neuralink = list()

neuralink_soc = soc_parameters.copy()
neuralink_soc["Name"] = "3"
neuralink_soc["active_channels"] = 1024
neuralink_soc["max_channels"] = 1024
neuralink_soc["sensing_area"] = 6.35 #mm^2
neuralink_soc["total_area"] = 20.48 #mm^2
neuralink_soc["power_consumption"] = 7.8 #2.8 #24.7 #mW # caclulcated relative power for sensing area from the total area of the chip
neuralink_soc["max_comm_channels"] = 1024
neuralink_soc["data_type"] = 10
neuralink_soc["sampling_period"] =  0.1 * 10**6 #10KHz
neuralink_soc["power_density_budget"] = 0.4 #mw/mm^2
neuralink_soc["budget_cutoff"] = [None, None] #to be determined

neuralink_comm_specs = communication.copy()
neuralink_comm_specs["energy_per_bit"] = 100 * 10**-12 #J
neuralink_comm_specs["max_data_rate"] = 50000 #J - a placeholder upper limit
neuralink_comm_specs["qam_efficiency"] = 15 #% - nominal
neuralink_comm_specs["qam_ber"] = 1*10**-6 # nominal
neuralink_comm_specs["qam_path_loss"] = 6 #60dB nominal
neuralink_comm_specs["qam_margin"] = 2 #20dB nominal

neuralink.append(neuralink_soc)
neuralink.append(neuralink_comm_specs)
socs.append(neuralink)


shen = list()

scale = 1024/16
area_scale = mt.sqrt(scale)#/16

shen_soc = soc_parameters.copy()
shen_soc["Name"] = "4"
shen_soc["active_channels"] = 16*scale
shen_soc["max_channels"] = 16*scale
shen_soc["sensing_area"] = 1.2 * 0.4*area_scale #mm^2
shen_soc["total_area"] = 1.344*area_scale #mm^2
shen_soc["power_consumption"] = 30.4/1000*scale#*scale #mW
shen_soc["max_comm_channels"] = 16*scale
shen_soc["data_type"] = 16
shen_soc["sampling_period"] =  0.1 * 10**6 #10KHz
shen_soc["power_density_budget"] = 0.4 #mw/mm^2
shen_soc["budget_cutoff"] = [None, None] #to be determined

shen_comm_specs = communication.copy()
shen_comm_specs["energy_per_bit"] = 100 * 10**-12 #J
shen_comm_specs["max_data_rate"] = 50000 #J - a placeholder upper limit
shen_comm_specs["qam_efficiency"] = 15 #% - nominal
shen_comm_specs["qam_ber"] = 1*10**-6 # nominal
shen_comm_specs["qam_path_loss"] = 6 #60dB nominal
shen_comm_specs["qam_margin"] = 2 #20dB nominal

shen.append(shen_soc)
shen.append(shen_comm_specs)
socs.append(shen)


muller = list()

scale = 1024/64
area_scale = mt.sqrt(scale)/2#/16

muller_soc = soc_parameters.copy()
muller_soc["Name"] = "5"
muller_soc["active_channels"] = 64*scale
muller_soc["max_channels"] = 64*scale
muller_soc["sensing_area"] = 1.6*area_scale #mm^2
muller_soc["total_area"] = 2.4 * 2.4*area_scale #mm^2
muller_soc["power_consumption"] = 147.2/1000*scale #mW
muller_soc["max_comm_channels"] = 64*scale
muller_soc["data_type"] = 15
muller_soc["sampling_period"] =  1 * 10**6 #1KHz
muller_soc["power_density_budget"] = 0.4 #mw/mm^2
muller_soc["budget_cutoff"] = [None, None] #to be determined

muller_comm_specs = communication.copy()
muller_comm_specs["energy_per_bit"] = 2.4 * 10**-12 #J
muller_comm_specs["max_data_rate"] = 50000 #J - a placeholder upper limit
muller_comm_specs["qam_efficiency"] = 15 #% - nominal
muller_comm_specs["qam_ber"] = 1*10**-6 # nominal
muller_comm_specs["qam_path_loss"] = 6 #60dB nominal
muller_comm_specs["qam_margin"] = 2 #20dB nominal

muller.append(muller_soc)
muller.append(muller_comm_specs)
socs.append(muller)


yang = list()

scale = 1024/4
area_scale = mt.sqrt(scale)#/16

yang_soc = soc_parameters.copy()
yang_soc["Name"] = "6"
yang_soc["active_channels"] = 4*scale
yang_soc["max_channels"] = 4*scale
yang_soc["sensing_area"] = 1.6*area_scale #mm^2
yang_soc["total_area"] = 2 * 2*area_scale #mm^2
yang_soc["power_consumption"] = 53.2/1000*scale #mW
yang_soc["max_comm_channels"] = 4*scale
yang_soc["data_type"] = 10
yang_soc["sampling_period"] =  1 * 10**6 / 20#20KHz
yang_soc["power_density_budget"] = 0.4 #mw/mm^2
yang_soc["budget_cutoff"] = [None, None] #to be determined

yang_comm_specs = communication.copy()
yang_comm_specs["energy_per_bit"] = 53.4 * 10**-12 #J
yang_comm_specs["max_data_rate"] = 50000 #J - a placeholder upper limit
yang_comm_specs["qam_efficiency"] = 15 #% - nominal
yang_comm_specs["qam_ber"] = 1*10**-6 # nominal
yang_comm_specs["qam_path_loss"] = 6 #60dB nominal
yang_comm_specs["qam_margin"] = 2 #20dB nominal

yang.append(yang_soc)
yang.append(yang_comm_specs)
socs.append(yang)


wimagine = list()

scale = 1024/64
area_scale = mt.sqrt(scale)/2#/16
volumetric_eff_scale = 50#100

wimagine_soc = soc_parameters.copy()
wimagine_soc["Name"] = "7"
wimagine_soc["active_channels"] = 64*scale
wimagine_soc["max_channels"] = 64*scale
wimagine_soc["sensing_area"] = 980*area_scale/volumetric_eff_scale #mm^2
wimagine_soc["total_area"] = 1960*area_scale/volumetric_eff_scale #mm^2
wimagine_soc["power_consumption"] = 75*scale/volumetric_eff_scale #mW
wimagine_soc["max_comm_channels"] = 64*scale
wimagine_soc["data_type"] = 12
wimagine_soc["sampling_period"] =  1 * 10**6 / 30#1KHz
wimagine_soc["power_density_budget"] = 0.4 #mw/mm^2
wimagine_soc["budget_cutoff"] = [None, None] #to be determined

wimagine_comm_specs = communication.copy()
wimagine_comm_specs["energy_per_bit"] = 50 * 10**-12 #J
wimagine_comm_specs["max_data_rate"] = 50000 #J - a placeholder upper limit
wimagine_comm_specs["qam_efficiency"] = 15 #% - nominal
wimagine_comm_specs["qam_ber"] = 1*10**-6 # nominal
wimagine_comm_specs["qam_path_loss"] = 6 #60dB nominal
wimagine_comm_specs["qam_margin"] = 2 #20dB nominal

wimagine.append(wimagine_soc)
wimagine.append(wimagine_comm_specs)
socs.append(wimagine)


halo = list()

scale = 1024/96
area_scale = 4*mt.sqrt(scale)*32 #technology to 45nm and a constant
volumetric_eff_scale = 8

halo_soc = soc_parameters.copy()
halo_soc["Name"] = "8"
halo_soc["active_channels"] = 96*scale
halo_soc["max_channels"] = 96*scale
halo_soc["sensing_area"] = 0.2*area_scale/volumetric_eff_scale #mm^2
halo_soc["total_area"] = 1*area_scale/volumetric_eff_scale #mm^2
halo_soc["power_consumption"] = 15*scale/volumetric_eff_scale #mW
halo_soc["max_comm_channels"] = 96*scale
halo_soc["data_type"] = 16
halo_soc["sampling_period"] =  1 * 10**6 / 30#30KHz
halo_soc["power_density_budget"] = 0.4 #mw/mm^2
halo_soc["budget_cutoff"] = [None, None] #to be determined

halo_comm_specs = communication.copy()
halo_comm_specs["energy_per_bit"] = 200 * 10**-12 #J
halo_comm_specs["max_data_rate"] = 50000 #J - a placeholder upper limit
halo_comm_specs["qam_efficiency"] = 15 #% - nominal
halo_comm_specs["qam_ber"] = 1*10**-6 # nominal
halo_comm_specs["qam_path_loss"] = 6 #60dB nominal
halo_comm_specs["qam_margin"] = 2 #20dB nominal

halo.append(halo_soc)
halo.append(halo_comm_specs)
socs.append(halo)

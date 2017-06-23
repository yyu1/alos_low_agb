#Generates a new AGB file based on input AGB and ALOS plus other classification images.
#All images are assumed to be the same size.

import numpy as np
import sys
import math

#ALOS function definitions
def trop_dry_broad( hv ):
	return 1.564*math.exp(77.591*hv)

def trop_shrub( hv ):
	return 1.564*math.exp(77.591*hv)

def med_woodland( hv ):
	return 272.97*math.pow(hv,0.41192)

def america_boreal( hv ):
	return 2755.1*math.pow(hv,1.2239)

def eurasia_boreal( hv ):
	return 3004.5*math.pow(hv,1.2693)

def apply_value( original_array, index, new_value ):
	nonzerocount = np.count_nonzero(index)
	if (nonzerocount == 0):
		return
	
	if (nonzerocount != new_value.size):
		print("Error when applying value, number of True in index does not equal number of new values, quiting")
		sys.exit()

	index_counter = 0
	for i in range(0,index.size):
		if (index[i]):
			if ((new_value[index_counter] < 50) and (original_array[i] < 50)):
				original_array[i] = math.floor(new_value[index_counter]*10+0.5)
			index_counter += 1
		


#Settings--------
xdim = 432000
ydim = 159600

#block_pixels = np.int64(ydim/100)*xdim
block_pixels = xdim

in_agb_file = '/dataraid/global/global_maxent_agb_combined_v6.int'
agb_type = np.int16
in_hv_file = '/dataraid/global/alos_2007_global_3sec_hv_cut_landsatfill.int'
hv_type = np.int16
in_globcover_file = '/dataraid/global/globcover_2006_mod100m_global.byt'
globcover_type = np.uint8
in_biome_file = '/dataraid/global/wwf_14biome_mod100m_global.byt'
biome_type = np.uint8

out_agb_file = '/dataraid/global/global_maxent_agb_combined_v6_alos_lowagb.int'

#----------------


#Open files for reading
fp_agb_in = open(in_agb_file, 'rb')
fp_hv_in = open(in_hv_file, 'rb')
fp_globcover_in = open(in_globcover_file, 'rb')
fp_biome_in = open(in_biome_file, 'rb')

#Open file for writing
fp_agb_out = open(out_agb_file, 'wb')


#vectorize functions
vfunc_trop_dry_broad = np.vectorize(trop_dry_broad, otypes=[np.float64])
vfunc_trop_shrub = np.vectorize(trop_shrub, otypes=[np.float64])
vfunc_med_woodland = np.vectorize(med_woodland, otypes=[np.float64])
vfunc_america_boreal = np.vectorize(america_boreal, otypes=[np.float64])
vfunc_eurasia_boreal = np.vectorize(eurasia_boreal, otypes=[np.float64])

#construct mask arrays for eurasia and north america for boreal
index_america = np.zeros(xdim,dtype=np.bool)
index_america[:index_america.size*140//360] = 1
index_eurasia = np.zeros(xdim,dtype=np.bool)
index_eurasia[index_eurasia.size*140//360:] = 1


for iBlock in range(0,ydim):
	agb_block = np.fromfile(fp_agb_in, dtype = agb_type, count = block_pixels)
	hv_block = np.fromfile(fp_hv_in, dtype = hv_type, count = block_pixels)
	globcover_block = np.fromfile(fp_globcover_in, dtype = globcover_type, count = block_pixels)
	biome_block = np.fromfile(fp_biome_in, dtype = biome_type, count = block_pixels)

	#-------Class masks
	index110 = (globcover_block == 110)
	index120 = (globcover_block == 120)
	index130 = (globcover_block == 130)
	index123 = np.logical_or(np.logical_or(index110,index120), index130)

	index7090 = np.logical_or((globcover_block == 70),(globcover_block == 90))

	index60 = (globcover_block == 60)   #woodlands
	index1236 = np.logical_or(index123,index60)

	index_biomeboreal = (biome_block == 6)
	index_biometropdrybroad = (biome_block == 2)
	index_biometropshrub = (biome_block == 7)
	index_biome_mediwoodland = (biome_block == 12)

	


	#-------Tropical Dry Broadleaf
	index = np.logical_and(index_biometropdrybroad, index1236)
	hv_agb = vfunc_trop_dry_broad(hv_block[index].astype(np.float64)/10000)
	apply_value(agb_block,index,hv_agb)

	#-------Tropical shrubland
	index = np.logical_and(index_biometropshrub, index123)
	hv_agb = vfunc_trop_shrub(hv_block[index].astype(np.float64)/10000)
	apply_value(agb_block,index,hv_agb)

	#-------Mediterranean Woodland
	index = np.logical_and(index_biome_mediwoodland,index1236)
	hv_agb = vfunc_med_woodland(hv_block[index].astype(np.float64)/10000)
	apply_value(agb_block,index,hv_agb)

	borealindex = np.logical_and(index_biomeboreal,index7090)
	#-------America Boreal
	index = np.logical_and(borealindex,index_america)
	hv_agb = vfunc_america_boreal(hv_block[index].astype(np.float64)/10000)
	apply_value(agb_block,index,hv_agb)

	#-------Eurasia Boreal
	index = np.logical_and(borealindex,index_eurasia)
	hv_agb = vfunc_eurasia_boreal(hv_block[index].astype(np.float64)/10000)
	apply_value(agb_block,index,hv_agb)

	#write out to file
	agb_block.tofile(fp_agb_out)



#close all open files
fp_agb_in.close()
fp_hv_in.close()
fp_globcover_in.close()
fp_biome_in.close()

fp_agb_out.close()


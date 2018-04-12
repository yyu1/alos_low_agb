#Generates a new AGB file based on input AGB and ALOS plus other classification images.
#All images are assumed to be the same size.

import numpy as np
import sys
import math
import numexpr

#ALOS function definitions
def trop_dry_broad( hv ):
	return numexpr.evaluate('1.564*exp(77.591*hv)')

def trop_shrub( hv ):
	return numexpr.evaluate('1.564*exp(77.591*hv)')

def med_woodland( hv ):
	return numexpr.evaluate('272.97*(hv**0.41192)')

def america_boreal( hv ):
	return numexpr.evaluate('2755.1*(hv**1.2239)')

def eurasia_boreal( hv ):
	return numexpr.evaluate('3004.5*(hv**1.2693)')

def asia_trop_moist(hv):
    return numexpr.evaluate('exp((1/0.25854)*log(hv/0.020863))')

def africa_trop_moist(hv):
    return numexpr.evaluate('7226.4 * (hv**1.2088)')

def america_trop_moist(hv):
    return numexpr.evaluate('exp((1/0.34334)*log(hv/0.015335))')

def temp_broad(hv):
    return numexpr.evaluate('418.62 * (hv**0.51193)')

def temp_conifer(hv):
    return numexpr.evaluate('exp((1/0.17064)*log(hv/0.033017))')

def fresh_flooded(hv):
    return numexpr.evaluate('1532.1 * (hv**0.92788)')

def saline_flooded(hv):
    return numexpr.evaluate('-4.4668 + 1120.9 * hv')

def apply_value(original_value, index, new_value):
    nonzerocount = np.count_nonzero(index)

    if (nonzerocount == 0):
        print("No non-zero in index, skipping block.")
        return
	
    if (nonzerocount != new_value.size):
        print("Error when applying value, number of True in index does not equal number of new values, quiting")
        sys.exit()

    replace_values = original_value[index]
    change_index = numexpr.evaluate('(replace_values < 500) & (new_value < 50)')

    new_value = numexpr.evaluate('new_value * 10 + 0.5')
    replace_values[change_index] = new_value[change_index]

    original_value[index] = replace_values



#Settings--------
xdim = 432000
ydim = 159600

block_pixels = np.int64(ydim//100)*xdim
#block_pixels = xdim

in_agb_file = '/dataraid/global/global_maxent_agb_combined_v9.int'
agb_type = np.int16
in_hv_file = '/dataraid/global/alos_2007_global_3sec_hv_cut_landsatfill.int'
hv_type = np.int16
in_globcover_file = '/dataraid/global/globcover_2006_mod100m_global.byt'
globcover_type = np.uint8
in_biome_file = '/dataraid/global/wwf_14biome_mod100m_global.byt'
biome_type = np.uint8
in_fnf_file = '/dataraid/global/alos_2015_global_3sec_fnf.byt'
fnf_type = np.uint8

out_agb_file = '/dataraid/global/global_maxent_agb_combined_v9_alos_lowagb_zeroed.int'

#----------------

#construct mask arrays for eurasia and north america for boreal
index_america = np.zeros((ydim//100,xdim),dtype=np.bool)
index_america[:,:xdim*140//360] = 1
index_eurasia = np.zeros((ydim//100,xdim),dtype=np.bool)
index_eurasia[:,xdim*140//360:] = 1

#construct mask arrays for tropical america, africa, and asia

index_trop_america = np.zeros((ydim//100,xdim),dtype=np.bool)
index_trop_america[:,:xdim*150//360] = 1

index_trop_africa = np.zeros((ydim//100,xdim),dtype=np.bool)
index_trop_africa[:,xdim*150//360:xdim*240//360] = 1

index_trop_asia = np.zeros((ydim//100,xdim),dtype=np.bool)
index_trop_asia[:,xdim*240//360:] = 1


def replace_low_value( agb_array, fnf_array, globcover_array, biome_array, hv_array):
    forest_index = numexpr.evaluate('(fnf_array > 0)')
    nonforest_index = np.logical_not(forest_index)
    more50index = numexpr.evaluate('(agb_array >= 500)')  #agb value is actual value * 10
    less50index = np.logical_not(more50index)

    # apply equation over all max<50, then set max > 50 and FNF = 0 to 0 this will be equivalent logic of:
    #  FNF > 0 and max < 50 -> apply equation
    #  FNF = 0 and max < 50 -> apply equation if there is equation
    #  FNF = 0 and max > 50 -> set = 0
    # But MAKE SURE max < 50 and > 50 index is obtained before zeroing FNF=0

    #create indices for landcover types
    index123crop = numexpr.evaluate('(globcover_array == 110) | (globcover_array == 120) | (globcover_array == 130) | (globcover_array == 20) | (globcover_array == 30)')
    index7090 = numexpr.evaluate('(globcover_array == 70) | (globcover_array == 90)')
    index1236crop = numexpr.evaluate('index123crop | (globcover_array == 60)')

    index7090 = numexpr.evaluate('(globcover_array == 70) | (globcover_array == 90)')

    index40 = numexpr.evaluate('(globcover_array == 40)')

    index_biomeboreal = numexpr.evaluate('(biome_array == 6)')
    index_biometropdrybroad = numexpr.evaluate('(biome_array == 2)')
    index_biometropshrub = numexpr.evaluate('(biome_array == 7)')
    index_biome_mediwoodland = numexpr.evaluate('(biome_array == 12)')

    #Tropical Dry Broadleaf
    index = numexpr.evaluate('(index1236crop & index_biometropdrybroad)')
    index_equation = index
    hv_agb = trop_dry_broad(hv_array[index].astype(np.float64)/10000)
    apply_value(agb_array, index, hv_agb)

    #Tropical Shrubland
    index = numexpr.evaluate('(index123crop & index_biometropshrub)')
    index_equation = numexpr.evaluate('(index_equation | index)')
    hv_agb = trop_shrub(hv_array[index].astype(np.float64)/10000)
    apply_value(agb_array, index, hv_agb)

    #Mediterranean Woodland
    index = numexpr.evaluate('(index1236crop & index_biome_mediwoodland)')
    index_equation = numexpr.evaluate('(index_equation | index)')
    hv_agb = med_woodland(hv_array[index].astype(np.float64)/10000)
    apply_value(agb_array, index, hv_agb)


    #boreal index
    index_boreal = numexpr.evaluate('(index_biomeboreal & index7090)')
    index_equation = numexpr.evaluate('(index_equation | index_boreal)')
    #American Boreal
    index = numexpr.evaluate('(index_boreal & index_america)')
    hv_agb = america_boreal(hv_array[index].astype(np.float64)/10000)
    apply_value(agb_array, index, hv_agb)
    #Eurasia Boreal
    index = numexpr.evaluate('(index_boreal & index_eurasia)')
    hv_agb = eurasia_boreal(hv_array[index].astype(np.float64)/10000)
    apply_value(agb_array, index, hv_agb)

    #Asia Tropical Moist
    index = numexpr.evaluate('index40 & index_trop_asia')
    index_equation = numexpr.evaluate('(index_equation | index)')
    hv_agb = asia_trop_moist(hv_array[index].astype(np.float64)/10000)
    apply_value(agb_array, index, hv_agb)

    #Africa Tropical Moist
    index = numexpr.evaluate('index40 & index_trop_africa')
    index_equation = numexpr.evaluate('(index_equation | index)')
    hv_agb = africa_trop_moist(hv_array[index].astype(np.float64)/10000)
    apply_value(agb_array, index, hv_agb)

    #America Tropical Moist
    index = numexpr.evaluate('index40 & index_trop_america')
    index_equation = numexpr.evaluate('(index_equation | index)')
    hv_agb = america_trop_moist(hv_array[index].astype(np.float64)/10000)
    apply_value(agb_array, index, hv_agb)

    #Temperate Broadleaf/Mixed
    index = numexpr.evaluate('((globcover_array == 50) | (globcover_array == 60)) & (biome_array == 4)')
    index_equation = numexpr.evaluate('(index_equation | index)')
    hv_agb = temp_broad(hv_array[index].astype(np.float64)/10000)
    apply_value(agb_array, index, hv_agb)

    #Temperate Conifer
    index = numexpr.evaluate('index7090 & (biome_array == 5)')
    index_equation = numexpr.evaluate('(index_equation | index)')
    hv_agb = temp_broad(hv_array[index].astype(np.float64)/10000)
    apply_value(agb_array, index, hv_agb)

    #Freshwater Flooded
    index = numexpr.evaluate('(globcover_array == 160)')
    index_equation = numexpr.evaluate('(index_equation | index)')
    hv_agb = temp_broad(hv_array[index].astype(np.float64)/10000)
    apply_value(agb_array, index, hv_agb)

    #Saline Flooded
    index = numexpr.evaluate('(globcover_array == 170)')
    index_equation = numexpr.evaluate('(index_equation | index)')
    hv_agb = temp_broad(hv_array[index].astype(np.float64)/10000)
    apply_value(agb_array, index, hv_agb)

    #Zero out FNF = 0 and Maxent > 50
    index = numexpr.evaluate('nonforest_index & more50index')
    agb_array[index] = 0

    #Zero out FNF = 0 and no equations
    index_noequation = np.logical_not(index_equation)
    index = numexpr.evaluate('nonforest_index & index_noequation')
    agb_array[index] = 0


#Open files for reading
fp_agb_in = open(in_agb_file, 'rb')
fp_hv_in = open(in_hv_file, 'rb')
fp_globcover_in = open(in_globcover_file, 'rb')
fp_biome_in = open(in_biome_file, 'rb')
fp_fnf_in = open(in_fnf_file, 'rb')

#Open file for writing
fp_agb_out = open(out_agb_file, 'wb')





for iBlock in range(0,100):
    print(iBlock)
    agb_block = np.fromfile(fp_agb_in, dtype = agb_type, count = block_pixels)
    agb_block.shape = (ydim//100,xdim)
    hv_block = np.fromfile(fp_hv_in, dtype = hv_type, count = block_pixels)
    hv_block.shape = (ydim//100,xdim)
    globcover_block = np.fromfile(fp_globcover_in, dtype = globcover_type, count = block_pixels)
    globcover_block.shape = (ydim//100,xdim)
    biome_block = np.fromfile(fp_biome_in, dtype = biome_type, count = block_pixels)
    biome_block.shape = (ydim//100,xdim)
    fnf_block = np.fromfile(fp_fnf_in, dtype = fnf_type, count = block_pixels)
    fnf_block.shape = (ydim//100,xdim) 
    replace_low_value(agb_block, fnf_block, globcover_block, biome_block, hv_block)
    
    #write out to file
    agb_block.tofile(fp_agb_out)



#close all open files
fp_agb_in.close()
fp_hv_in.close()
fp_globcover_in.close()
fp_biome_in.close()

fp_agb_out.close()


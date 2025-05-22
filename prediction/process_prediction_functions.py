import numpy as np
import scipy
import xarray as xr
import os 
import pandas as pd
import multiprocessing
import pickle
import sklearn
    
def load_config(model_path):
    with open(os.path.join(model_path,'config.pkl'),'rb') as f:
        config_run = pickle.load(f)
    config_run['regional_var'] = ['CO','BC','OC','NH3','NO','SO2','VOC']
    
    return(config_run)

def convert_to_area_total_emis(input_xr, config, area_xr):
    
    output_xr = input_xr.copy().load()
    # Compute gridcell area if precomputed area file i    
    for v in config['regional_var']:
        output_xr[v] *= area_xr
    
    return(output_xr)

def geographic_weighting(input_xr, model_path, config):
    with open(os.path.join(model_path,'sigma.pkl'),'rb') as f:
        sigma_species = pickle.load(f)
    
    output_xr = input_xr.copy().load()

    for i,v in enumerate(config['regional_var']):
        output_xr[v].values = scipy.ndimage.gaussian_filter(input_xr[v],sigma = (0,sigma_species[i],sigma_species[i]), truncate=sigma_species[i]*10)
    
    return(output_xr)

def normalization(input_xr, model_path):
    output_xr = input_xr.copy().load()

    with open(os.path.join(model_path,'pm25_var.pkl'),'rb') as f:
        pm25_var = pickle.load(f)

    with open(os.path.join(model_path,'pm25_norm_coef.pkl'),'rb') as f:
        pm25_norm_coef = pickle.load(f)
        
    for i,v in enumerate(pm25_var):
        output_xr[v] /= pm25_norm_coef[i]

    return(output_xr)

def input_to_ndarray(input_xr,model_path):
    
    output_xr = input_xr.to_array(dim='species').load()

    ## find the species-to-array-index mapping
    with open(os.path.join(model_path,'pm25_var.pkl'),'rb') as f:
        pm25_var = pickle.load(f)

    pm25_index = []

    for v in pm25_var:
        pm25_index.append(np.where(output_xr.species == v)[0][0])
    
    return({'ndarray_in':output_xr,
           'pm25_index':pm25_index})

def preprocess_input(input_xr, model_path, do_area_aggregation = True, co2_level = 397.2, ch4_level = 1830.795):
        
    ### Load config as basic data
    config = load_config(model_path)
    area_xr = xr.open_dataarray(os.path.join(model_path,'area_2x25.nc'))
    
    ### Check if all variable exists
    input_xr_vars = list(input_xr.variables)
    with open(os.path.join(model_path,'pm25_var.pkl'),'rb') as f:
        pm25_var = pickle.load(f)
    
    all_vars_exist = True
    missing_var_list = []
    
    ### fill the missing fields
    for v in pm25_var:            
        if v not in input_xr_vars:
            if v == 'CO2':
                fill_xr = (xr.ones_like(area_xr) * co2_level).rename(v)
            else:
                if v == 'CH4':
                    fill_xr = (xr.ones_like(area_xr) * ch4_level).rename(v)
                else:
                    ### If emission fields missing for particular species, assume no emission change
                    fill_xr = xr.zeros_like(area_xr).rename(v)
                        
            input_xr[v] = fill_xr
            
    ### Now do all preprocessing of input data
    if do_area_aggregation:
        area_aggregated_emis_xr = convert_to_area_total_emis(input_xr, config, area_xr)
    else:
        area_aggregated_emis_xr = input_xr.copy()
    
    if config['gaussian_blur']:
        area_aggregated_emis_gw_xr = geographic_weighting(area_aggregated_emis_xr, model_path, config)
    else:
        area_aggregated_emis_gw_xr = area_aggregated_emis_xr.copy()

    emis_gw_normalized_xr = normalization(area_aggregated_emis_gw_xr, model_path)
    gw_gpr_input = input_to_ndarray(emis_gw_normalized_xr, model_path)     
    
    return(gw_gpr_input)

def do_prediction_gp(input_xr, pm25_index, model_path):
    
    with open(os.path.join(model_path,'gp_pm25.pkl'), 'rb') as f:
        gp_pm25_list = pickle.load(f)

    with open(os.path.join(model_path,'coord_list.pkl'),'rb') as f:
        ij_list = pickle.load(f)
        i_list = ij_list[0]
        j_list = ij_list[1]
    
    ## Get un-normalization indicies
    mean_y = xr.open_dataset(os.path.join(model_path,'mean_xy.nc'))['dPM25']
    abs_y = xr.open_dataset(os.path.join(model_path,'abs_xy.nc'))['dPM25']
    
    ## produce output array 
    dpm25_gp = xr.zeros_like(input_xr[0,...]).rename('dPM25')
    dpm25_gp_sigma = xr.zeros_like(dpm25_gp).rename('sigma_dPM25')

    ## now do prediction
    for k in range(len(i_list)):
        i = i_list[k]
        j = j_list[k]
        x_pm25_local = input_xr[pm25_index,:,j,i].values.transpose()
        dpm25_gp_local = gp_pm25_list[k].predict(x_pm25_local,return_std=True) #do standard deviation
        dpm25_gp[:,j,i] = dpm25_gp_local[0] * abs_y[j,i].values + mean_y[j,i].values
        dpm25_gp_sigma[:,j,i] = dpm25_gp_local[1] * abs_y[j,i].values
        
    return({'mean': dpm25_gp,
        'std': dpm25_gp_sigma,
    })
                     

def do_prediction_lin(input_xr, pm25_index, model_path):
    
    with open(os.path.join(model_path,'lin_pm25.pkl'), 'rb') as f:
        lin_pm25_list = pickle.load(f)

    with open(os.path.join(model_path,'coord_list.pkl'),'rb') as f:
        ij_list = pickle.load(f)
        i_list = ij_list[0]
        j_list = ij_list[1]
    
    ## Get un-normalization indicies
    mean_y = xr.open_dataset(os.path.join(model_path,'mean_xy.nc'))['dPM25']
    abs_y = xr.open_dataset(os.path.join(model_path,'abs_xy.nc'))['dPM25']
    
    ## produce output array 
    dpm25_lin = xr.zeros_like(input_xr[0,...]).rename('dPM25')

    ## now do prediction
    for k in range(len(i_list)):
        i = i_list[k]
        j = j_list[k]
        x_pm25_local = input_xr[pm25_index,:,j,i].values.transpose()
        dpm25_lin[:,j,i] = lin_pm25_list[k].predict(x_pm25_local) * abs_y[j,i].values + mean_y[j,i].values 
    
    return(dpm25_lin)


def get_dpm25(input_xr, model_path, mode = 'gp'):
    gw_gpr_input = preprocess_input(input_xr, model_path)
    if mode == 'gp':
        output = do_prediction_gp(gw_gpr_input['ndarray_in'], gw_gpr_input['pm25_index'], model_path)
    if mode == 'lin':
        output = do_prediction_lin(gw_gpr_input['ndarray_in'], gw_gpr_input['pm25_index'], model_path)

    return(output)



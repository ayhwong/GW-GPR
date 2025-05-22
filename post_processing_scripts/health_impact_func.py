import numpy as np
from scipy.stats import norm
import pandas as pd
import numba

def get_gemm_param(gemm_crf_table,p_value = 0.05):

    ### Define CRF
    gemm_param = pd.read_csv(gemm_crf_table)
    gemm_ncd_lri = gemm_param.loc[(gemm_param['cause'] == 'NCD+LRI') & (gemm_param['age'] != '>25')]
    gemm_ncd_lri_with_key = gemm_ncd_lri.set_index('age')

    ### If optimize aggressively
    gemm_theta_vec = np.array(gemm_ncd_lri_with_key['theta'])

    ### For CI calculation
    gemm_theta_vec_low = np.zeros_like(gemm_theta_vec)
    gemm_theta_vec_high = np.zeros_like(gemm_theta_vec)

    for i in range(len(gemm_theta_vec)):
        gemm_param = gemm_ncd_lri.iloc[i]
        gemm_theta_vec_low[i] = norm.ppf(p_value/2,loc = gemm_param['theta'],scale = gemm_param['std theta'])
        gemm_theta_vec_high[i] = norm.ppf(1-p_value/2,loc = gemm_param['theta'],scale = gemm_param['std theta'])
    
    return({
        'mean':gemm_theta_vec,
        'low' : gemm_theta_vec_low,
        'high' : gemm_theta_vec_high
    })


@numba.jit()
def get_dmort_pm25(pm25_base,pm25_counterfactual,ncd_lri_death_array,theta):
    
    dmort_pm25 = np.zeros_like(pm25_base)
    affected_population = ncd_lri_death_array.sum(axis=0)

    for i in range(dmort_pm25.shape[1]):
        for j in range(dmort_pm25.shape[0]):
            ### skip when population = 0
            if np.isnan(affected_population[j,i]) or affected_population[j,i] == 0:
                continue
                
            pm25_i = pm25_counterfactual[j,i]
            pm25_0 = pm25_base[j,i]
                        
            for k in range(ncd_lri_death_array.shape[0]):
                
                if pm25_0 < 2.4:
                    rr_0 = 1.0
                else:
                    rr_0 = np.exp(theta[k] * np.log(1 + pm25_0/1.6) * (1/(1+np.exp(-(pm25_0-15.5)/36.8))))
                
                if pm25_i < 2.4:
                    rr_i = 1.0
                else:
                    rr_i = np.exp(theta[k] * np.log(1 + pm25_i/1.6) * (1/(1+np.exp(-(pm25_i-15.5)/36.8))))

                rr_frac = rr_i/rr_0 - 1
                dmort_pm25[j,i] += ncd_lri_death_array[k,j,i] * rr_frac
                
    return dmort_pm25
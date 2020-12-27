def BARC_calculator(data,lambda_,theta,fname=None) :
    import numpy as np
    import pandas as pd
    
    def optical_coeff(N,trigonometric_ftn_ij) :
        # precalculate the optical coefficients
        n0_cos_0 = N[0]*trigonometric_ftn_ij[0,1]
        n1_cos_1 = N[1]*trigonometric_ftn_ij[1,1]
        n0_cos_1 = N[0]*trigonometric_ftn_ij[1,1]
        n1_cos_0 = N[1]*trigonometric_ftn_ij[0,1]
        reflection_coeff_s = (n0_cos_0-n1_cos_1)/(n0_cos_0+n1_cos_1)
        transmission_coeff_s = 2.*n0_cos_0/(n0_cos_0+n1_cos_1)
        reflection_coeff_p = (n1_cos_0-n0_cos_1)/(n1_cos_0+n0_cos_1)
        transmission_coeff_p = 2.*n0_cos_0/(n1_cos_0+n0_cos_1)      
        return np.array([reflection_coeff_s, transmission_coeff_s,
                         reflection_coeff_p, transmission_coeff_p])
        
    def I_matrix(optical_coeff_i) :
        # reflection and transmission
        I = np.ones([2,2], dtype=complex)
        I[[0,1],[1,0]] *= optical_coeff_i[0]
        I /= optical_coeff_i[1]
        return I

    def L_matrix(N_i,trigonometric_ftn_i,d,lambda_) :
        # absorption
        beta = 2.*np.pi*d/lambda_*N_i*trigonometric_ftn_i[1]
        L = np.zeros([2,2], dtype=complex)
        L[0,0], L[1,1] = np.exp(1j*beta), np.exp(-1j*beta)
        return L

    layers = data['layer'][0].astype(int)
    N, trigonometric_ftn = np.ones(layers+2, dtype=complex), np.empty([layers+2,2], dtype=complex)
    N[1:-1] = np.array(data['n']-data['k']*1.j)

    # Calculate sine values
    trigonometric_ftn[0,0] = np.sin(theta)
    for i in range(1,layers+2) : trigonometric_ftn[i,0] = N[i-1]/N[i]*trigonometric_ftn[i-1,0]
    # Calculate cosine values
    trigonometric_ftn[:,1] = (1.-trigonometric_ftn[:,0]*trigonometric_ftn[:,0].conjugate())**0.5

    nm_ = 1e-9
    lambda_ *= nm_
    d = np.zeros(layers+2)
    d[1:-1] = np.array(data['d'])*nm_
    
    # calculate the S matrices
    S_s = I_matrix(optical_coeff(N[:2],trigonometric_ftn[:2])[:2])
    S_p = I_matrix(optical_coeff(N[:2],trigonometric_ftn[:2])[2:])
    for i in range(1, layers+1) :
        S_s = np.dot(S_s, L_matrix(N[i],trigonometric_ftn[i],d[i],lambda_))
        S_s = np.dot(S_s, I_matrix(optical_coeff(N[i:i+2],trigonometric_ftn[i:i+2])[:2]))
        S_p = np.dot(S_p, L_matrix(N[i],trigonometric_ftn[i],d[i],lambda_))
        S_p = np.dot(S_p, I_matrix(optical_coeff(N[i:i+2],trigonometric_ftn[i:i+2])[2:]))
        
    # the output of TE_r, TE_t, TM_r, TM_t
    results = np.array([S_s[1,0]/S_s[0,0], 1./S_s[0,0], S_p[1,0]/S_p[0,0], 1./S_p[0,0]])
    results = np.abs(results)**2*100
    
    if fname!=None : 
#         print('append at', fname)
        f=open(fname,'a')
        np.savetxt(f, [np.concatenate((results, np.array(data), np.degrees(theta)), axis=None)], delimiter=',', fmt='%.5e')
        f.close()
    return results
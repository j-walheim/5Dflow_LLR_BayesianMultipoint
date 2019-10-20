import numpy as np
import scipy.optimize as spoptimize
from numba import jit


def BayesUnfold(img, kv):
    # input:
    # -img: image data_: [Nx Ny Nz Nhp Nseg]:
    #  segment order: [ref,
    #         kv_x1,kv_y1,kv_z1,kv_x2,kv_y2,kv_z2]
    # -kv: velocity encodings [kv_x1,0,0; 0,kv_y1,0; ...]
    # out:
    #    -img: mean image magnitude
    #    -results: mean velocities per voxel
    #    -reults_D: intra-voxel standard deviations (i.e. sigma, not sigma^2)



    noDirs = np.size(kv, 1)
    print('start bayes unfolding, input image size: ', img.shape,' \t number of directions: ',noDirs)

    results_v = np.zeros((np.size(img, 0), np.size(img, 1), np.size(img, 2), np.size(img, 3), noDirs))
    results_std = np.zeros((np.size(img, 0), np.size(img, 1), np.size(img, 2), np.size(img, 3), noDirs))

    img = img * np.conj(img[:, :, :, :, 0, None])



    for indDir in range(noDirs):
        kv_full = kv

        kv_curr = kv[:, indDir]

        no_pts = np.sum(kv_curr != 0)  # get number of points
        print('no_pts  ', no_pts)



        ind_data = np.array(0)
        for i in range(no_pts):
            ind_data = np.append(ind_data, 1+indDir+i*noDirs)
        print('ind_data', ind_data)
        
        venc = np.pi / np.min(np.abs(kv_curr[kv_curr != 0]));

        for indext in range(np.size(img, 3)):
            print('Direction: ', indDir, ' max VENC: ', venc)
            print('Timepoint: ', indext, ' out of ', np.size(img, 3))

            for indexz in range(np.size(img, 2)):
                sdata = img[:, :, indexz, indext, ind_data]

                xdim = np.size(sdata, 0)
                ydim = np.size(sdata, 1)

                sdata = np.reshape(sdata, (np.prod(np.size(sdata[:, :, 0])), np.size(sdata, 2)))
                sdata = np.transpose(sdata)

                # calculate vresults, vresults_D
                vresults, vresults_std = solve_TKE_all(sdata, kv_curr)

                results_v[:, :, indexz, indext, indDir] = np.reshape(vresults, (xdim, ydim))
                results_std[:, :, indexz, indext, indDir] = np.reshape(vresults_std, (xdim, ydim))

    return results_v,results_std


def solve_TKE(sdata, kv, verbose=0):

    VSTEP_SCALE = 30 # define grid size for initial v estimate

    Nkv = np.size(sdata,0)

    D_est = estimate_sigma(sdata, kv, Nkv)

    # Estimate vel
    venc = np.pi / np.min(kv[kv > 0]);  # Venc is from the smallest non-zero kv point
    vstep = np.pi / np.max(kv[kv > 0]) / VSTEP_SCALE;  # // Step size is in reference to the smallest VENC
    v_est = estimate_vel(venc, vstep, D_est, sdata, kv, Nkv);

    # estimate v_est, D_est for Nelder-Mead search
    x0 = np.array([v_est, D_est])

    if np.isnan(D_est):
        D_est = 0.

    # define probfun as function of x (= v, sigma) with
    probfun = lambda x: postprob(x, sdata, kv, Nkv)

    # search maximum probability (-log..), with starting point x0
    # output: v_out, D_out
    out = spoptimize.minimize(probfun, x0, method='Nelder-Mead', options={'disp': False, 'maxiter':50})
    v_out = out.x[0]
    D_out = out.x[1]

    return v_out, D_out

def solve_TKE_all(sdata, kv):

    Nkv = np.size(sdata, 0)
    N = np.size(sdata, 1)

    vresults_D = np.zeros(N)
    vresults = np.zeros(N)

    for i in range(N):
        v_out, D_out = solve_TKE(sdata[:, i], kv, 0)
        vresults[i] = v_out
        vresults_D[i] = D_out;
    return vresults, vresults_D

@jit(nopython = True)
def postprob(x_in, sdata, kv, Ns):
    sdata_r = np.real(sdata)
    sdata_i = np.imag(sdata)

    v = x_in[0]
    sig = x_in[1]

    m = 2
    sigsq = sig * sig

    eig1 = 0.0;
    for i in range(Ns):
        eig1 = eig1 + np.exp(-sigsq * kv[i] * kv[i])
    eig1 = 1 / eig1;

    h1 = 0.0;
    h2 = 0.0;
    for i in range(Ns):
        h1 = h1 + np.exp(-0.5 * sigsq * kv[i] * kv[i]) * (
        sdata_r[i] * np.sin(v * kv[i]) + sdata_i[i] * np.cos(v * kv[i]));
        h2 = h2 + np.exp(-0.5 * sigsq * kv[i] * kv[i]) * (
        sdata_r[i] * np.cos(v * kv[i]) - sdata_i[i] * np.sin(v * kv[i]));

    h1 = h1 * np.sqrt(eig1)
    h2 = h2 * np.sqrt(eig1)

    mh2 = h1 * h1 + h2 * h2

    denom = 0.0
    for i in range(Ns):
        denom = denom + sdata_r[i] * sdata_r[i] + sdata_i[i] * sdata_i[i];

    if(denom==0.0):
        denom = 1e-10

    compprob = -np.log(eig1 * (1 - mh2 / denom) ** ((m - 2 * Ns) / 2));

    return compprob

@jit(nopython = True)
def estimate_vel(venc, vstep, D_est, sdata, kv, Nkv):

    xin = [-venc, D_est];

    min_tprob = postprob(xin, sdata, kv, Nkv)
    v_est = -venc;
    tprob = 0.0;

    for v in np.arange(-venc, venc, vstep):
        xin[0] = v;
        tprob = postprob(xin, sdata, kv, Nkv)
        if (tprob < min_tprob):
            min_tprob = tprob
            v_est = v

    return v_est

#@jit(nopython = True)
def estimate_sigma(sdata, kv, Nkv):
    # /** Estimate sigma for a voxel
    norm = 0.0
    for i in range(Nkv):
        norm = norm + (0.5 * kv[i] * kv[i]) ** 2

    sig = 0.0;
    y0 = np.abs(sdata[0])

    y = 0.0;
    for i in range(Nkv):
        y = np.log(y0 / (abs(sdata[i]) + 1e-16 ))
        if (y < 0):
            y = 0.0

        sig = 0.5 * kv[i] ** 2 * y / norm + sig

    sig = np.sqrt(np.abs(sig))
    return  sig

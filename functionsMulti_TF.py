import numpy as np
import scipy.optimize as spoptimize
import tensorflow as tf
from numba import jit
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1' #make TF select the right GPU

def BayesUnfold(img, kv):
    # Bayes unfolding according to model presented in Binter et al. (2014): 'Bayesian multipoint velocity encoding
    # for concurrent flow and turbulence mapping.
    # input:
    #    -img: image data; segment order: [ref,
    #       kv_x1,kv_y1,kv_z1,kv_x2,kv_y2,kv_z2]
    #    -kv: [kv_x1,0,0; 0,kv_y1,0; ...]
    # out:
    #    -img: mean image magnitude
    #    -results: velocities(unwrapped)
    #    -reults_D: intra-voxel standard deviations (i.e. sigma, not sigma^2)



    noDirs = np.size(kv, 1)
    print('start bayes unfolding, input image size: ', img.shape,' \t number of directions: ',noDirs)

    results_v = np.zeros((np.size(img, 0), np.size(img, 1), np.size(img, 2), np.size(img, 3), noDirs))
    results_std = np.zeros((np.size(img, 0), np.size(img, 1), np.size(img, 2), np.size(img, 3), noDirs))

    img = img * np.conj(img[:, :, :, :, 0, None]) #subtract background phase

    #perform Bayes unfolding separately for each direction
    for indDir in range(noDirs):

        kv_curr = kv[:, indDir]

        no_pts = np.sum(kv_curr != 0)  # get number of points
        print('no_pts  ', no_pts)

        ind_data = np.array(0)
        for i in range(no_pts):
            ind_data = np.append(ind_data, 1+indDir+i*noDirs)
        print('ind_data', ind_data)


        sdata = img[:, :, :, :, ind_data]
        szim = np.shape(img[...,0])

        sdata = np.reshape(sdata, (np.prod(np.size(sdata[...,0])),-1))
        sdata = np.transpose(sdata)

        vresults, vresults_std = solve_TKE_TF(sdata, kv_curr)

        results_v[..., indDir] = np.reshape(vresults, szim)
        results_std[... , indDir] = np.reshape(vresults_std, szim)


    return results_v,results_std


def solve_TKE_TF(sdata, kv):

    VSTEP_SCALE = 30#0 # define grid size for initial v estimate

    Nkv = np.size(sdata,0)

    D_est = estimate_sigma(sdata, kv, Nkv)

    print('type',   type(D_est))
    print('shape', D_est.shape)

    # Estimate vel
    venc = np.pi / np.min(kv[kv > 0]);  # Venc is from the smallest non-zero kv point
    vstep = np.pi / np.max(kv[kv > 0]) / VSTEP_SCALE;  # // Step size is in reference to the smallest VENC
    v_est = estimate_vel(venc, vstep, D_est, sdata, kv, Nkv);

    # estimate v_est, D_est for Nelder-Mead search
    x0 = np.array([v_est, D_est])

    D_est[np.isnan(D_est)] = 0.0


    #tensorflow starts here
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
    config.allow_soft_placement = True  # If a operation is not define it the default device, let it execute in another.
    sess = tf.InteractiveSession(config=config)

    # Set up TensorFlow variables
    tf_x = tf.Variable(x0, dtype=tf.float32, trainable=True)
    tf_kv = tf.constant(kv, dtype=tf.float32)
    tf_sdata = tf.constant(sdata, dtype=tf.complex64)


    #objective function: posterior probability of x given the measured data
    fval = tf.reduce_sum(postprob_TF(tf_x, tf_sdata, tf_kv, Nkv))


    # Set up optimizer
    lr = 0.1
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gvs = optimizer.compute_gradients(fval)

    #train_op = optimizer.apply_gradients(gvs)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
    train_op = optimizer.apply_gradients(capped_gvs)

    sess.run(tf.global_variables_initializer())

    #number of iterations and learning rate are not really tuned yet. There might be quite some potential for optimization.
    for it in range(2000):
        feed = {learning_rate: lr, tf_kv : kv, tf_sdata: sdata}
#        if (it + 1) % 20 == 0:
#            lr = lr / 1.5
        if it % 10 == 0:
            c_err = fval.eval(feed_dict=feed)
            print('iter: ', it, ' ____ ', c_err)

        train_op.run(feed_dict=feed)

    x = tf_x.eval()

    sess.close()

    return x[0], x[1]



def postprob_TF(x_in, sdata, kv, Ns):
    sdata_r = tf.real(sdata)
    sdata_i = tf.imag(sdata)


    v = x_in[0,:]
    sig = x_in[1,:]

    m = 2
    sigsq = sig * sig

    eig1 = 0.0;
    for i in range(Ns):
        eig1 = eig1 + tf.math.exp(-sigsq * kv[i] * kv[i])
    eig1 = 1 / eig1;

    h1 = 0.0 * sigsq;
    h2 = 0.0 * sigsq;
    
    for i in range(Ns):
        h1 = h1 + tf.exp(-0.5 * sigsq * kv[i] * kv[i]) * (
            sdata_r[i,:] * tf.math.sin(v * kv[i]) + sdata_i[i,:] * tf.math.cos(v * kv[i]));
        h2 = h2 + tf.exp(-0.5 * sigsq * kv[i] * kv[i]) * (
            sdata_r[i,:] * tf.math.cos(v * kv[i]) - sdata_i[i,:] * tf.math.sin(v * kv[i]));

    h1 = h1 * tf.sqrt(eig1)
    h2 = h2 * tf.sqrt(eig1)

    mh2 = h1 * h1 + h2 * h2

    denom = 0.0

    for i in range(Ns):
        denom = denom + sdata_r[i] * sdata_r[i] + sdata_i[i] * sdata_i[i];

    denom = denom + 1e-10* (denom == 0)
    prob = eig1 * (1 - mh2 / denom) ** ((m - 2 * Ns) / 2)

    compprob = -tf.log(prob)

    return compprob

def postprob(x_in, sdata, kv, Ns):
    sdata_r = np.real(sdata)
    sdata_i = np.imag(sdata)

    v = x_in[0, :]
    sig = x_in[1, :]

    m = 2
    sigsq = sig * sig

    eig1 = 0.0;
    for i in range(Ns):
        eig1 = eig1 + np.exp(-sigsq * kv[i] * kv[i])
    eig1 = 1 / eig1;

    h1 = 0.0 * sigsq;
    h2 = 0.0 * sigsq;

    for i in range(Ns):
        h1 = h1 + np.exp(-0.5 * sigsq * kv[i] * kv[i]) * (
            sdata_r[i, :] * np.sin(v * kv[i]) + sdata_i[i, :] * np.cos(v * kv[i]));
        h2 = h2 + np.exp(-0.5 * sigsq * kv[i] * kv[i]) * (
            sdata_r[i, :] * np.cos(v * kv[i]) - sdata_i[i, :] * np.sin(v * kv[i]));

    h1 = h1 * np.sqrt(eig1)
    h2 = h2 * np.sqrt(eig1)
    mh2 = h1 * h1 + h2 * h2

    denom = 0.0

    for i in range(Ns):
        denom = denom + sdata_r[i] * sdata_r[i] + sdata_i[i] * sdata_i[i];

    denom = denom + 1e-10* (denom == 0)

    compprob = -np.log(eig1 * (1 - mh2 / denom) ** ((m - 2 * Ns) / 2));

    return compprob


def estimate_vel(venc, vstep, D_est, sdata, kv, Nkv):

    xin = np.array([-venc* np.ones_like(D_est), D_est]);

    min_tprob = postprob(xin, sdata, kv, Nkv)
    v_est = -venc* np.ones_like(D_est)
    tprob = 0.0*np.ones_like(D_est);

    for v in np.arange(-venc, venc, vstep):
        xin[0,:] = v;
        tprob = postprob(xin, sdata, kv, Nkv)
  
        ind_tmp = (tprob < min_tprob)

        v_est[ind_tmp] = np.repeat(v,np.sum(ind_tmp))
        min_tprob[ind_tmp] = tprob[ind_tmp]

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
        y = np.log(y0 / (np.abs(sdata[i]) + 1e-16 ))

        y[y<0] = 0.0

        sig = 0.5 * kv[i] ** 2 * y / norm + sig

    sig = np.sqrt(np.abs(sig))
    return  sig



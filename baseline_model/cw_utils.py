#!~/anaconda2/envs/ipython3/bin/python

import numpy as np
import scipy as sp
import scipy.stats as stats
import pandas as pd
import copy
import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pickle


def N_A(**kwargs):
    N = kwargs['N'] if 'N' in kwargs else 5
    A = kwargs['A'] if 'A' in kwargs else np.eye(N)
    return N, A

def KalmanFilterUpdate(y_t, x_tmin1_tmin1, P_tmin1_tmin1, Phi, Q, R, **kwargs):

    weight = kwargs['weight'] if 'weight' in kwargs else 1
    N, A = N_A(**kwargs)

    # ii - we need the forward Kalman Filter
    x_t_tmin1 = Phi.dot(x_tmin1_tmin1)
    P_t_tmin1 = Phi.dot(P_tmin1_tmin1).dot(Phi.T)
    P_t_tmin1[0:N, 0:N] += Q

    sigma_t = A.dot(P_t_tmin1).dot(A.T) + R
    sigma_t_inv = np.linalg.pinv(sigma_t)

    K_t = P_t_tmin1.dot(A.T).dot(sigma_t_inv)

    epsilon_t = (y_t - (A.dot(x_t_tmin1))) * weight

    x_t_t = x_t_tmin1 + K_t.dot(epsilon_t)
    P_t_t = (np.eye(N) - K_t.dot(A)).dot(P_t_tmin1)

    return {
        "x_t_t" : x_t_t,
        "x_tmin1_t" : x_t_tmin1,
        "P_t_t" : P_t_t,
        "P_tmin_t" : P_t_tmin1,
        "Sig_t_inv" : sigma_t_inv,
        "Sig_t" : sigma_t,
        "Innovation_t" : epsilon_t,
        "K_t": K_t
        }


def ForwardKalmanFilter(ys, mu0, Sig0, Phi, Q, R, **kwargs):

    x_tmin1_tmin1 = mu0
    P_tmin1_tmin1 = Sig0

    Xs = np.zeros((len(ys) + 1, len(x_tmin1_tmin1)))
    Ps = np.zeros((len(ys) + 1, P_tmin1_tmin1.shape[0], P_tmin1_tmin1.shape[1]))
    X_smin1_s = np.zeros((len(ys) + 1, len(x_tmin1_tmin1)))
    P_smin1_s = np.zeros((len(ys) + 1, P_tmin1_tmin1.shape[0], P_tmin1_tmin1.shape[1]))


    Xs[0] = x_tmin1_tmin1
    Ps[0] = P_tmin1_tmin1
    X_smin1_s[0] = x_tmin1_tmin1
    P_smin1_s[0] = P_tmin1_tmin1

    incomplete_likelihood = 0
    innovations = np.zeros_like(ys)

    for i, yt in enumerate(ys):

        if 'weights' in kwargs:

            param_update = KalmanFilterUpdate(
                y_t = yt,
                x_tmin1_tmin1 = Xs[i],
                P_tmin1_tmin1 = Ps[i],
                Phi = Phi,
                Q = Q,
                R = R,
                N = kwargs['N'],
                weight = kwargs['weights'][i]
            )
        else:
            param_update = KalmanFilterUpdate(
                y_t = yt,
                x_tmin1_tmin1 = Xs[i],
                P_tmin1_tmin1 = Ps[i],
                Phi = Phi,
                Q = Q,
                R = R,
                N = kwargs['N']
            )

        x_t_t = param_update['x_t_t']
        P_t_t = param_update['P_t_t']

        x_tmin1_t = param_update['x_tmin1_t']
        P_tmin1_t = param_update['P_tmin_t']

        Xs[i+1] = x_t_t
        Ps[i+1] = P_t_t
        X_smin1_s[i+1] = x_tmin1_t
        P_smin1_s[i+1] = P_tmin1_t

        this_likelihood = 0.5 * np.log(np.linalg.det(param_update['Sig_t']))
        innov_err = param_update['Innovation_t'].T.dot(param_update['Sig_t_inv']).dot(param_update['Innovation_t'])
        this_likelihood += 0.5 * np.log(innov_err)

        incomplete_likelihood += this_likelihood

        innovations[i] = param_update['Innovation_t']

    return {
        "Xs": Xs,
        "Ps": Ps,
        "X_smin1_s":X_smin1_s,
        "P_smin1_s":P_smin1_s,
        "Kn": param_update['K_t'],
        "incomplete_likelihood": incomplete_likelihood,
        'innovations': innovations
    }

def SmootherUpdate(x_n_t, P_n_t, t, Phi, fwd_params, **kwargs):

    x_tmin1_tmin1 = fwd_params['Xs'][t-1]
    x_tmin1_t = fwd_params['X_smin1_s'][t]

    P_tmin1_tmin1 = fwd_params['Ps'][t-1]
    P_tmin1_t = fwd_params['P_smin1_s'][t]

    J_tmin1 = P_tmin1_tmin1.dot(Phi.T).dot(np.linalg.pinv(P_tmin1_t))

    weight = 1 if 'weights' not in kwargs else kwargs['weights'][t]

    x_n_tmin1 = x_tmin1_tmin1 + J_tmin1.dot(x_n_t - x_tmin1_t)
    P_n_tmin1 = P_tmin1_tmin1 + J_tmin1.dot( P_n_t - P_tmin1_t).dot(J_tmin1.T)

    return {
        "x_n_tmin1": x_n_tmin1,
        "P_n_tmin1": P_n_tmin1,
        "J_tmin1" : J_tmin1
    }

def KalmanSmoother(fwd_params, Phi, **kwargs):

    n = len(fwd_params["Xs"])

    x_n_n = fwd_params["Xs"][-1]
    P_n_n = fwd_params["Ps"][-1]

    Xn = np.zeros((n, len(x_n_n)))
    Pn = np.zeros((n, P_n_n.shape[0], P_n_n.shape[1]))
    Jn = np.zeros((n, P_n_n.shape[0], P_n_n.shape[1]))

    Xn[-1] = x_n_n
    Pn[-1] = P_n_n

    for t in range(n-1, -1, -1):

        smt_update = SmootherUpdate(x_n_n, P_n_n, t, Phi, fwd_params, **kwargs)

        x_n_n = smt_update["x_n_tmin1"]
        P_n_n = smt_update["P_n_tmin1"]

        Xn[t] = x_n_n
        Pn[t] = P_n_n
        Jn[t] = smt_update["J_tmin1"]

    return {
        "Xn": Xn,
        "Pn": Pn,
        "Jn": Jn
    }

def OneLagUpdate(P_n_t_tmin1, t, Ps, Jn, Phi):

    P_n_tmin1_tmin2 = Ps[t-1].dot(Jn[t-2].T) + Jn[t-1].dot(P_n_t_tmin1 - Phi.dot(Ps[t-1])).dot(Jn[t-2].T)

    return P_n_tmin1_tmin2

def OneLagCovarianceSmoother(fwd_params, bkwd_params, Phi, **kwargs):

    N, A = N_A(**kwargs)

    n = len(fwd_params["Xs"])
    K = len(fwd_params["Xs"][0])

    Kn = fwd_params['Kn']
    P_n_n_nmin1 = (np.eye(K) - Kn.dot(A)).dot(Phi).dot(bkwd_params["Pn"][-2])

    P_one_lag = np.zeros((n, P_n_n_nmin1.shape[0], P_n_n_nmin1.shape[1]))
    P_one_lag[-1] = P_n_n_nmin1

    Ps = fwd_params['Ps']
    Jn = bkwd_params['Jn']

    for t in range(n-1, 1, -1):

        P_n_tmin1_tmin2 = OneLagUpdate(P_n_n_nmin1, t, Ps, Jn, Phi)
        P_one_lag[t] = P_n_tmin1_tmin2

    return P_one_lag

def MaximisationUpdate(ys, fwd_params, bkwd_params, Phi, **kwargs):

    N, A = N_A(**kwargs)

    Pn = bkwd_params['Pn']
    Xn = bkwd_params['Xn']
    Pn_one_lag = OneLagCovarianceSmoother(fwd_params, bkwd_params, Phi, **kwargs)

    n = len(fwd_params['Xs'])
    start = 2
    end = n

    S_1_0 = np.zeros_like(fwd_params['Ps'][0])
    S_1_1 = np.zeros_like(fwd_params['Ps'][0])
    S_0_0 = np.zeros_like(fwd_params['Ps'][0])

    for i, (x_n_t, x_n_tmin1, P_n_t, P_n_tmin1, P_n_t_tmin1) in enumerate(zip(Xn[start+1:end],
                                                                Xn[start:end-1],
                                                                Pn[start+1:end],
                                                                Pn[start:end-1],
                                                                Pn_one_lag[start:end-1])):

        x_n_t = x_n_t.reshape(-1,1)
        x_n_tmin1 = x_n_tmin1.reshape(-1,1)

        S_1_1 += x_n_t.dot(x_n_t.T) + P_n_t
        S_1_0 += x_n_t.dot(x_n_tmin1.T) + P_n_t_tmin1
        S_0_0 += x_n_tmin1.dot(x_n_tmin1.T) + P_n_tmin1

    reg_ = 5e-2
    S_0_0_inv = np.linalg.pinv(S_0_0 + reg_*np.eye(N))

    Phi_j = (S_1_0+reg_*np.eye(N)).dot(S_0_0_inv)
    Q_j = 1/(n) * (S_1_1 - S_1_0.dot(S_0_0_inv).dot(S_1_0.T))

    for i, row in enumerate(Q_j):
        for j,_ in enumerate(row):
            if i != j:
                Q_j[i,j] = 0

    R_j = np.zeros_like(A.dot(A.T))
    for y, X, P in zip(ys[1:], Xn[1:], Pn[1:]):

        R_j += (y - A.dot(X)).dot((y - A.dot(X)).T) + A.dot(P).dot(A.T)

    R_j /= (n)

    return { "mu0": Xn[1],
             "muEnd": Xn[-1],
             "Sig0": Pn[1],
             "Phi": Phi_j,
             "Q": Q_j,
             "R": R_j }

def EM_step(ys, params, **kwargs):

    mu0 = params['mu0']
    Sig0 = params['Sig0']
    Phi = params['Phi']
    Q = params['Q']
    R = params['R']

    # Expectation Step (using Kalman Filter and Smoother)
    fwd_params = ForwardKalmanFilter(ys, mu0, Sig0, Phi, Q, R, **kwargs)
    bkwd_params = KalmanSmoother(fwd_params, Phi)

    # Maximisation Step (using update equation that was derived in [1])
    update_params = MaximisationUpdate(ys, fwd_params, bkwd_params, Phi, **kwargs)

    return update_params, fwd_params

def ExpectationMaximisation(ys, starting_params, **kwargs):

    num_iter = 100

    params = starting_params
    lklihoods = np.zeros(num_iter)

    for k in range(num_iter):

        params_new, fwd_params = EM_step(ys, params, **kwargs)

        Phi = params_new['Phi']

        params['Phi'] = Phi
        params['mu0'] = params_new['mu0']
        params['Sig0'] = params_new['Sig0']
        params['muEnd'] = params_new['muEnd']
        params['Q'] = params_new['Q']
        params['innovations'] = fwd_params['innovations']
#         params['R'] = params_new['R']

        lklihoods[k] = fwd_params['incomplete_likelihood']

        if k == 0:
            continue

        if np.linalg.norm(lklihoods[k] - lklihoods[k-1]) < 1e-2:
            break

    return params, lklihoods

def learn_breakpoint(ys, num_breaks, starting_params, percentage=98, sensitivity=2):

    T = len(ys)
    N = len(ys[0])
    break_points = []

    # initialise the cut points
    breaks = np.concatenate([[0], np.arange(T//num_breaks, T, T//num_breaks)[:num_breaks-1], [T]])
    break_points.append(breaks)

    # iterate to find the new breaks
    penalty = 0
    for k in range(20):

        print('starting', breaks)
        result_params, fwd_filts = [], []

        try:
            new_breaks = []
            for j in range(1, len(breaks)):

                p_, _ = ExpectationMaximisation(ys[breaks[j-1]:breaks[j]], copy.deepcopy(starting_params), N=starting_params['N'])
                result_params.append(p_)

                # fwd_filt = ForwardKalmanFilter(
                #             ys,
                #             p_['mu0'],
                #             p_['Sig0'],
                #             p_['Phi'],
                #             p_['Q'],
                #             p_['R'],
                #             N=starting_params['N']
                #         )
                #
                # # alpha = 3
                # # beta = (3*2/len(breaks-1))*j
                # # log_prior = np.log(1/T)
                # prob = stats.multivariate_normal(mean=np.zeros(starting_params['N']), cov=1e-4*np.eye(starting_params['N'])).logpdf(fwd_filt['innovations'])
                # errs.append(prob.reshape(-1,1)

            for j in range(1, len(breaks)-1):

                lp_l = np.zeros(T-1)
                lp_r = np.zeros(T-1)

                R = np.eye(N) * 1e-3
                phi1 = result_params[j-1]['Phi']
                phi2 = result_params[j]['Phi']

                ts = np.arange(0,1,1/T)

                alpha = j*1/(len(breaks)-1) * num_breaks + 1

                # alpha = breaks[j]/T * num_breaks + 1
                beta = num_breaks + 2 - alpha

                # the 4 here means we have a prior on the number of chains that we expect
                tot = alpha + beta
                alpha = alpha/tot * (2+num_breaks)
                beta = beta/tot * (2+num_breaks)

                lp_l = np.zeros(T-1)
                lp_r = np.zeros(T-1)

                for t in range(1, T-1):
                    lp_l[t] = lp_l[t-1] + stats.multivariate_normal(np.dot(phi1, ys[t-1]), R).logpdf(ys[t])
                    lp_r[t] = lp_r[t-1] + stats.multivariate_normal(np.dot(phi2, ys[t-1]), R).logpdf(ys[t])

                lp = stats.beta(alpha, beta).logpdf(ts[1:-1]) + np.ones(T-2)*lp_r[-1] + lp_l[:-1] - lp_r[:-1]
                # lp = -np.log(T) + np.ones(T-2)*lp_r[-1] + lp_l[:-1] - lp_r[:-1]
                # lp = np.ones(T-2)*np.log(1/T) + np.ones(T-2)*lp_r[-1] + lp_l[:-1] - lp_r[:-1]

                new_break = np.argmax(lp) + 1
                new_breaks.append(new_break)

            if num_breaks > 1:
                ## TODO: Seriously you need to make a better boundary decision than this.
                breaks = [0, T] + new_breaks
                # for i in range(1, num_breaks):
                #     penalty += sensitivity
                #     if penalty > 8:
                #         penalty = 8
                #     # how heavily are we penalising outliers?
                #     max_ = np.sort(np.where(np.argmax(errs, axis=1) == i-1))[0][-(penalty+1)]
                #     min_ = np.sort(np.where(np.argmax(errs, axis=1) == i))[0][penalty]
                #     # print('updating:', min_, max_)
                #     breaks.append((max_ + min_)//2)
                breaks = np.array(np.sort(breaks))
                print('updated', breaks)

            else:
                print('only two breaks')
                breaks = np.concatenate([[0],[T]])

            if len(np.where(np.diff(breaks) < 30)[0]) > 0:

                to_rem = np.where(np.diff(breaks) < 30)[0][0] + 1
                breaks = list(breaks)
                del breaks[to_rem]
                print("**chain collapse", to_rem)
                num_breaks -= 1
                # breaks = np.array(breaks)
                # num_breaks -= 1
                # breaks = np.concatenate([[0], np.arange(T//num_breaks, T, T//num_breaks)[:num_breaks-1], [T]])
                break_points.append(breaks)
                continue

            if len(breaks) == len(break_points[-1]):
                print('checking convergence', k)
                if np.sum(np.abs((breaks - break_points[-1]))) <= 4:
                    print('Converged')
                    break

            break_points.append(breaks)

        except:

            num_breaks -= 1
            breaks = np.concatenate([[0], np.arange(T//num_breaks, T, T//num_breaks)[:num_breaks-1], [T]])
            break_points.append(breaks)
            print("**SVD Converge Caught", num_breaks)
            continue

    return {
        'result_params': result_params,
        'fwd_filts': fwd_filts,
        'breaks': breaks
    }

def update_img_time(n, cap, im, ann):

    ret,frame = cap.read()

    min_ = n//60
    sec_ = n - min_*60

    ann.set_text("frame: %i:%02d min" % (min_, sec_))

    if ret==True:
        im.set_data(frame[864:1080,0:384])

    return im

def get_mini_view_video(fname_mov, fname_csv):

    cap = cv2.VideoCapture(fname_mov)
    count = 0

    dpi = 100

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ret,frame = cap.read()

    im = ax.imshow(frame[864:1080,0:384])
    fig.set_size_inches([5,5])
    ann = ax.annotate("frame: %i:%02d min" % (0, 0), xy=[0,0], xytext=[0,0])

    ani = animation.FuncAnimation(fig, update_img_time, len(pd.read_csv(fname_csv)), fargs=(cap, im, ann), interval=500)
    writer = animation.writers['ffmpeg'](fps=10)

    ani.save(fname_mov.replace('.mov', '_cleaned.mov'), writer=writer)

    return True

def plot_data_and_boundaries(df, em_results, ax, **kwargs):

    ys = df.values
    times = em_results['breaks'][1:]
    times[-1] = len(ys)
    results = em_results['result_params']

    N, A = N_A(**kwargs)

    if 'water' in kwargs:
        water = kwargs['water']
    else:
        water = [np.zeros(N)]
        water[0][0] = 1

    timesteps = np.arange(1,len(ys))

    Phi = results[0]['Phi']

    j = 1
    for i in timesteps:

        if j <= len(times):
            if i > times[j-1]:

                Phi = results[j]['Phi']
                j += 1

        water.append(A.dot(Phi.dot(water[i-1])))

    water = np.array(water)

    c = ['b', 'g', 'r', 'purple', 'gold', 'pink']
    labels=['waterfall', 'desert', 'plains', 'jungle', 'wetlands', 'reservoir']

    for i in range(len(water[0])):
        ax.plot(water[:,i], c=c[i], ls='--')

    for j in times[:-1]:
        ax.axvline(j, c='black')
        min_ = j//60
        sec_ = (j - min_*60)
        ax.annotate("%i:%02d min" % (min_, sec_), xy=(j, 1), xytext=(j-20, 1.1),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

    df.plot(ax=ax, color=c)
    ax.set_xticklabels([round(i/60,1) for i in ax.get_xticks()])

    ax.set_ylabel('Normalised Water Level')
    ax.set_xlabel('Time (min)')
    return ax


def stan_sampled(df, M=4, sw=[]):
    T = df.shape[0]
    N = df.shape[1]

    # initialise the switches uniformly
    if len(sw) <= 0:
        sw = np.array([i for i in range(0,T,T//(M+1))][:-1] + [T])

    with open('./../stan/known_switches_unknown_params.pkl', 'rb') as f:
        ssm = pickle.load(f)
        data = {
            'y': df.values,
            'N': N,
            'T': T,
            'M': len(sw)-1,
            'S': sw[1:]
        }
        fit = ssm.sampling(data=data, iter=2000, chains=4)

    la = fit.extract(permuted=True)
    phis = la.get('phi').mean(axis=0)

    fits = []
    with open('./../stan/unknown_switches_known_params.pkl', 'rb') as f:
        ssm = pickle.load(f)

        for i in range(len(sw)-2):
            data = {'y': df.values,
                'N': N,
                'T': T,
                'M': 2,
                'phi': phis[i:i+2,:],
                'sw': sw[i+1],
                'p': 15 }

            fits.append(ssm.sampling(data=data, iter=2000, chains=4, seed=12))

    new_splits = set()
    for fit in fits:

        la = fit.extract(permuted=True)
        means = np.mean(la.get('lp'), axis=0)
        new_max = np.argmax(means)
        print(new_max)
        if new_max < 20:
            new_max = 0
        elif new_max > T-20:
            new_max = T

        if not (np.abs(np.array(list(new_splits)) - new_max) < 15).any():
            new_splits.add(new_max)

    new_splits.add(0)
    new_splits.add(T)

    new_splits = np.sort(list(new_splits))

    sw = new_splits

    with open('./../stan/known_switches_unknown_params.pkl', 'rb') as f:
        ssm = pickle.load(f)
        data = {
            'y': df.values,
            'N': N,
            'T': T,
            'M': len(sw)-1,
            'S': sw[1:]
        }
        fit = ssm.sampling(data=data, iter=2000, chains=4)

    la = fit.extract(permuted=True)
    stan_results = {}
    phis = la.get('phi').mean(axis=0)

    return {
        'breaks':  np.sort(sw),
        'result_params': [{'Phi':p} for p in la.get('phi').mean(axis=0)],
        'fit': fit
    }

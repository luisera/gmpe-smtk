#!/usr/bin/env/python

'''
General Class for extracting Ground Motion Intensity Measures (IMs) from a
set of acceleration time series
'''

import numpy as np
from math import pi
from scipy.integrate import cumtrapz
from scipy.stats import scoreatpercentile
import smtk.response_spectrum as rsp

RESP_METHOD = {'Newmark-Beta': rsp.NewmarkBeta,
               'Nigam-Jennings': rsp.NigamJennings}


def get_velocity_displacement(time_step, acceleration):
    '''
    Returns the velocity and displacment time series using simple integration
    :param float time_step:
        Time-series time-step (s)
    :param numpy.ndarray acceleration:
        Acceleration time-history
    :returns:
        velocity - Velocity Time series (cm/s)
        displacement - Displacement Time series (cm)
    '''
    velocity = time_step * cumtrapz(acceleration, initial=0.)
    displacement = time_step * cumtrapz(velocity, initial=0.)
    return velocity, displacement


def get_peak_measures(time_step, acceleration, get_vel=False, 
    get_disp=False):
    '''

    '''
    pga = np.max(np.fabs(acceleration))
    if get_disp:
        get_vel = True
    if get_vel:
        velocity = time_step * cumtrapz(acceleration, initial=0.)
        pgv = np.max(np.fabs(velocity))
    else:
        pgv = None
    if get_disp:
        displacement = time_step * cumtrapz(velocity, initial=0.)
        pgd = np.max(np.fabs(displacement))
    else:
        pgd = None
    return pga, pgv, pgd, velocity, displacement

def get_response_spectrum(acceleration, time_step, periods, damping=0.05, 
        units="cm/s/s", method="Nigam-Jennings"):
    '''
    Returns the response spectrum
    '''
    response_spec = RESP_METHOD[method](acceleration,
                                        time_step,
                                        periods, 
                                        damping,
                                        units)
    spectrum, time_series, accel, vel, disp = response_spec.evaluate()

    return spectrum, time_series, accel, vel, disp


def get_response_spectrum_pair(acceleration_x, time_step_x, acceleration_y,
        time_step_y, periods, damping=0.05, units="cm/s/s",
        method="Nigam-Jennings"):
    '''
    Returns the response spectrum
    '''

    sax = get_response_spectrum(acceleration_x,
                                time_step_x,
                                periods,
                                damping, 
                                units, 
                                method)[0]
    sax["PGA"] = np.max(np.fabs(acceleration_x))
    say = get_response_spectrum(acceleration_y,
                                time_step_y,
                                periods, 
                                damping, 
                                units, 
                                method)[0]
    say["PGA"] = np.max(np.fabs(acceleration_y))
    return sax, say


#def geometric_mean_spectrum(acceleration_x, time_step_x, acceleration_y,
#        time_step_y, periods, damping=0.05, units="cm/s/s",
#        method="Nigam-Jennings"):
#    '''
#    Returns the geometric mean of the response spectrum
#    '''
#    sax, say = get_response_spectrum_pair(acceleration_x, time_step_x,
#                                          acceleration_y, time_step_y,
#                                          periods, damping, units, method)

def geometric_mean_spectrum(sax, say):
    """
    Returns the geometric mean of the response spectrum
    """
    sa_gm = {}
    for key in sax.keys():
        if key == "Period":
            sa_gm[key] = sax[key]
        else:
            sa_gm[key] = np.sqrt(sax[key] * say[key])
    return sa_gm


def arithmetic_mean_spectrum(sax, say):
    """
    Returns the arithmetic mean of the response spectrum
    """
    sa_am = {}
    for key in sax.keys():
        if key == "Period":
            sa_am[key] = sax[key]
        else:
            sa_am[key] = (sax[key] + say[key]) / 2.0
    return sa_am

        
def envelope_spectrum(sax, say):
    """
    Returns the envelope of the response spectrum
    """
    sa_env = {}
    for key in sax.keys():
        if key == "Period":
            sa_env[key] = sax[key]
        else:
            sa_env[key] = np.max(np.column_stack([sax[key], say[key]]),
                                 axis=1)
    return sa_env

def larger_pga(sax, say):
    """

    """
    if sax["PGA"] >= say["PGA"]:
        return sax
    else:
        return say


def rotate_horizontal(series_x, series_y, angle):
    """
    Rotates two time-series according to the angle
    """
    angle = angle * (pi / 180.0)
    rot_hist_x = (np.cos(angle) * series_x) + (np.sin(angle) * series_y)
    rot_hist_y = (-np.sin(angle) * series_x) + (np.cos(angle) * series_y)
    return rot_hist_x, rot_hist_y

def equalise_series(series_x, series_y):
    """
    """
    n_x = len(series_x)
    n_y = len(series_y)
    if n_x > n_y:
        return series_x[:n_y], series_y
    elif n_y > n_x:
        return series_x, series_y[:n_x]
    else:
        return series_x, series_y

def gmrotdpp(acceleration_x, time_step_x, acceleration_y, time_step_y, periods,
        percentile, damping=0.05, units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the rotationally-dependent geometric mean
    """
    if (percentile > 100. + 1E-9) or (percentile < 0.):
        raise ValueError("Percentile for GMRotDpp must be between 0. and 100.")
    # Get the time-series corresponding to the SDOF
    sax, _, x_a, _, _ = get_response_spectrum(acceleration_x,
                                              time_step_x,
                                              periods, damping,
                                              units, method)
    say, _, y_a, _, _ = get_response_spectrum(acceleration_y,
                                              time_step_y,
                                              periods, damping,
                                              units, method)
    x_a, y_a = equalise_series(x_a, y_a)
    angles = np.arange(0., 90., 1.)
    max_a_theta = np.zeros([len(angles), len(periods)], dtype=float)
    max_a_theta[0, :] = np.sqrt(np.max(np.fabs(x_a), axis=0) *
                                np.max(np.fabs(y_a), axis=0))
    for iloc, theta in enumerate(angles):
        if iloc == 0:
            max_a_theta[iloc, :] = np.sqrt(np.max(np.fabs(x_a), axis=0) *
                                           np.max(np.fabs(y_a), axis=0))
        else:
            rot_x, rot_y = rotate_horizontal(x_a, y_a, theta)
            max_a_theta[iloc, :] = np.sqrt(np.max(np.fabs(rot_x), axis=0) *
                                           np.max(np.fabs(rot_y), axis=0))
    return scoreatpercentile(max_a_theta, percentile), max_a_theta, angles

def _get_gmrotd_penalty(gmrotd, gmtheta):
    """
    
    """
    n_angles, n_per = np.shape(gmtheta)
    penalty = np.zeros(n_angles, dtype=float)
    coeff = 1. / float(n_per)
    for iloc in range(0, n_angles):
        penalty[iloc] = coeff * np.sum(((gmtheta[iloc] / gmrotd) - 1.) ** 2.)

    locn = np.argmin(penalty)
    return locn, penalty


def gmrotipp(acceleration_x, time_step_x, acceleration_y, time_step_y, periods,
        percentile, damping=0.05, units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the rotationally-independent geometric mean (GMRotIpp)
    """
    acceleration_x, acceleration_y = equalise_series(acceleration_x,
                                                     acceleration_y)
    gmrotd, gmtheta, angle = gmrotdpp(acceleration_x, time_step_x,
                                      acceleration_y, time_step_y, 
                                      periods, percentile, damping, units, 
                                      method)
    
    min_loc, penalty = _get_gmrotd_penalty(gmrotd, gmtheta)
    target_angle = angle[min_loc]

    rot_hist_x, rot_hist_y = rotate_horizontal(acceleration_x,
                                               acceleration_y,
                                               target_angle)
    sax, say = get_response_spectrum_pair(rot_hist_x, time_step_x,
                                          rot_hist_y, time_step_y,
                                          periods, damping, units, method)

    return geometric_mean_spectrum(sax, say)

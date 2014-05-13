#!/usr/bin/env/python

"""
Module to get GMPE residuals - total, inter and intra
{'GMPE': {'IMT1': {'Total': [], 'Inter event': [], 'Intra event': []},
          'IMT2': { ... }}}
          


"""
import h5py
import numpy as np
from math import sqrt
from scipy.special import erf
from scipy.stats import scoreatpercentile, norm
from copy import deepcopy
from collections import OrderedDict
from openquake.hazardlib.gsim import get_available_gsims
import smtk.intensity_measures as ims
from openquake.hazardlib import imt
from smtk.strong_motion_selector import SMRecordSelector


GSIM_LIST = get_available_gsims()
GSIM_KEYS = set(GSIM_LIST.keys())

#SCALAR_IMTS = ["PGA", "PGV", "PGD", "CAV", "Ia"]
SCALAR_IMTS = ["PGA", "PGV"]


def get_interpolated_period(target_period, periods, values):
    """
    Returns the spectra interpolated in loglog space
    :param float target_period:
        Period required for interpolation
    :param np.ndarray periods:
        Spectral Periods
    :param np.ndarray values:
        Ground motion values
    """
    if (target_period < np.min(periods)) or (target_period > np.max(periods)):
        return None, "Period not within calculated range %s"
    lval = np.where(periods <= target_period)[0][-1]
    uval = np.where(periods >= target_period)[0][0]
    if (uval - lval) == 0:
        return values[lval]
    else:
        dy = np.log10(values[uval]) - np.log10(values[lval])
        dx = np.log10(periods[uval]) - np.log10(periods[lval])
        
        return 10.0 ** (
            np.log10(values[lval]) + 
            (np.log10(target_period) - np.log10(periods[lval])) * dy / dx
            )


def get_geometric_mean(fle):
    """
    Retreive geometric mean of the ground motions from the file - or calculate
    if not in file
    :param fle:
        Instance of :class: h5py.File
    """
    #periods = fle["IMS/X/Spectra/Response/Periods"].value
    if not "H" in fle["IMS"].keys():
        # Horizontal spectra not in record
        x_spc = fle["IMS/X/Spectra/Response/Acceleration/damping_05"].values
        y_spc = fle["IMS/Y/Spectra/Response/Acceleration/damping_05"].values
        periods = fle["IMS/X/Spectra/Response/Periods"].values
        sa_geom = np.sqrt(x_spc * y_spc)
    else:
        if "Geometric" in fle["IMS/H/Spectra/Response/Acceleration"].keys():
            sa_geom =fle[
                "IMS/H/Spectra/Response/Acceleration/Geometric/damping_05"
                ].value
            periods = fle["IMS/X/Spectra/Periods"].values
            idx = periods > 0
            periods = periods[idx]
            sa_geom = sa_geom[idx]
        else:
            # Horizontal spectra not in record
            x_spc = fle[
                "IMS/X/Spectra/Response/Acceleration/damping_05"].values
            y_spc = fle[
                "IMS/Y/Spectra/Response/Acceleration/damping_05"].values
            sa_geom = np.sqrt(x_spc * y_spc)
    return sa_geom

def get_gmrotd50(fle):
    """
    Retrieve GMRotD50 from file (or calculate if not present)
    :param fle:
        Instance of :class: h5py.File
    """
    periods = fle["IMS/X/Spectra/Response/Periods"].value
    periods = periods[periods > 0.]
    if not "H" in fle["IMS"].keys():
        # Horizontal spectra not in record
        x_acc = ["Time Series/X/Original Record/Acceleration"]
        y_acc = ["Time Series/Y/Original Record/Acceleration"]
        sa_gmrotd50 = ims.gmrotdpp(x_acc.value, x_acc.attrs["Time-step"],
                                   y_acc.value, y_acc.attrs["Time-step"],
                                   periods, 50.0)[0]
        
    else:
        if "GMRotD50" in fle["IMS/H/Spectra/Response/Acceleration"].keys():
            sa_gmrotd50 =fle[
                "IMS/H/Spectra/Response/Acceleration/GMRotD50/damping_05"
                ].value
        else:
            # Horizontal spectra not in record - calculate from time series
            x_acc = ["Time Series/X/Original Record/Acceleration"]
            y_acc = ["Time Series/Y/Original Record/Acceleration"]
            sa_gmrotd50 = ims.gmrotdpp(x_acc.value, x_acc.attrs["Time-step"],
                                       y_acc.value, y_acc.attrs["Time-step"],
                                       periods, 50.0)[0]
    return sa_gmrotd50

def get_gmroti50(fle):
    """   
    Retreive GMRotI50 from file (or calculate if not present)
    :param fle:
        Instance of :class: h5py.File
    """
    periods = fle["IMS/X/Spectra/Response/Periods"].value
    periods = periods[periods > 0.]
    if not "H" in fle["IMS"].keys():
        # Horizontal spectra not in record
        x_acc = ["Time Series/X/Original Record/Acceleration"]
        y_acc = ["Time Series/Y/Original Record/Acceleration"]
        sa_gmroti50 = ims.gmrotipp(x_acc.value, x_acc.attrs["Time-step"],
                                   y_acc.value, y_acc.attrs["Time-step"],
                                   periods, 50.0)[0]
        
    else:
        if "GMRotI50" in fle["IMS/H/Spectra/Response/Acceleration"].keys():
            sa_gmroti50 =fle[
                "IMS/H/Spectra/Response/Acceleration/GMRotI50/damping_05"
                ].value
        else:
            # Horizontal spectra not in record - calculate from time series
            x_acc = ["Time Series/X/Original Record/Acceleration"]
            y_acc = ["Time Series/Y/Original Record/Acceleration"]
            sa_gmroti50 = ims.gmrotipp(x_acc.value, x_acc.attrs["Time-step"],
                                       y_acc.value, y_acc.attrs["Time-step"],
                                       periods, 50.0)
            # Assumes Psuedo-spectral acceleration
            sa_gmroti50 = sa_gmroti50["PSA"]
    return sa_gmroti50

SPECTRA_FROM_FILE = {"Geometric": get_geometric_mean,
                     "GMRotI50": get_gmroti50,
                     "GMRotD50": get_gmrotd50}


SCALAR_XY = {"Geometric": lambda x, y : np.sqrt(x * y),
             "Arithmetic": lambda x, y : (x + y) / 2.,
             "Larger": lambda x, y: np.max(np.array([x, y])),
             "Vectorial": lambda x, y : np.sqrt(x ** 2. + y ** 2.)}

def get_scalar(fle, i_m, component="Geometric"):
    """
    Retrieves the scalar IM from the database
    :param fle:
        Instance of :class: h5py.File
    :param str i_m:
        Intensity measure
    :param str component:
        Horizontal component of IM
    """
     
    if not "H" in fle["IMS"].keys():
        x_im = fle["IMS/X/Scalar/" + i_m].value[0]
        y_im = fle["IMS/Y/Scalar/" + i_m].value[0]
        return SCALAR_XY[component](x_im, y_im)
    else:
        if i_m in fle["IMS/H/Scalar"].keys():
            return fle["IMS/H/Scalar/" + i_m].value[0]
        else:
            raise ValueError("Scalar IM %s not in record database" % i_m)



class Residuals(object):
    """
    Class to derive sets of residuals for a list of ground motion residuals
    according to the GMPEs
    """
    def __init__(self, gmpe_list, imts):
        """
        :param list gmpe_list:
            List of GMPE names (using the standard openquake strings)
        :param list imts:
            List of Intensity Measures
        """
        self.gmpe_list = gmpe_list
        self.number_gmpes = len(self.gmpe_list)
        self.types = OrderedDict([(gmpe, {}) for gmpe in self.gmpe_list])
        self.residuals = []
        self.imts = imts
        for gmpe in gmpe_list:
            if not gmpe in GSIM_LIST:
                raise ValueError("%s not supported in OpenQuake" % gmpe) 
            gmpe_dict = {}
            for imtx in self.imts:
                gmpe_dict[imtx] = {}
                self.types[gmpe][imtx] = []
                for res_type in \
                    GSIM_LIST[gmpe].DEFINED_FOR_STANDARD_DEVIATION_TYPES:
                    gmpe_dict[imtx][res_type] = []
                    self.types[gmpe][imtx].append(res_type)
            self.residuals.append([gmpe, gmpe_dict])
        self.residuals = OrderedDict(self.residuals)
        self.database = None
        self.number_records = None
        self.contexts = None
    

    def get_residuals(self, database, nodal_plane_index=1,
            component="Geometric"):
        """
        Calculate the residuals for a set of ground motion records
        """
        # Contexts is a list of dictionaries 
        contexts = database.get_contexts(nodal_plane_index)
        self.database = SMRecordSelector(database)
        self.contexts = []
        for context in contexts:
            #print context
            # Get the observed strong ground motions
            context = self.get_observations(context, component)
            # Get the expected ground motions
            context = self.get_expected_motions(context)
            context = self.calculate_residuals(context)
            for gmpe in self.residuals.keys():
                for imtx in self.residuals[gmpe].keys():
                    for res_type in self.residuals[gmpe][imtx].keys():
                        self.residuals[gmpe][imtx][res_type].extend(
                            context["Residual"][gmpe][imtx][res_type].tolist())
            self.contexts.append(context)
       
        for gmpe in self.residuals.keys():
            for imtx in self.residuals[gmpe].keys():
                for res_type in self.residuals[gmpe][imtx].keys():
                    self.residuals[gmpe][imtx][res_type] = np.array(
                        self.residuals[gmpe][imtx][res_type])

    def get_observations(self, context, component="Geometric"):
        """
        Get the obsered ground motions from the database
        """
        select_records = self.database.select_from_event_id(context["EventID"])
        observations = OrderedDict([(imtx, []) for imtx in self.imts])
        selection_string = "IMS/H/Spectra/Response/Acceleration/"
        for record in select_records:
            fle = h5py.File(record.datafile, "r")
            for imtx in self.imts:
                if imtx in SCALAR_IMTS:
                    if imtx == "PGA":
                        observations[imtx].append(
                            get_scalar(fle, imtx, component) / 981.0)
                    else:
                        observations[imtx].append(
                            get_scalar(fle, imtx, component))

                elif "SA(" in imtx:
                    target_period = imt.from_string(imtx).period
                    
                    spectrum = fle[selection_string + component 
                                   + "/damping_05"].value
                    periods = fle["IMS/H/Spectra/Response/Periods"].value
                    observations[imtx].append(get_interpolated_period(
                        target_period, periods, spectrum) / 981.0)
                else:
                    raise "IMT %s is unsupported!" % imtx
            fle.close()
        for imtx in self.imts:
            observations[imtx] = np.array(observations[imtx])
        context["Observations"] = observations
        return context

    def get_expected_motions(self, context):
        """
        Calculate the expected ground motions from the context
        """
        # TODO Rake hack will be removed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if not context["Rupture"].rake:
            context["Rupture"].rake = 0.0
        expected = OrderedDict([(gmpe, {}) for gmpe in self.gmpe_list])
        for gmpe in self.gmpe_list:
            expected[gmpe] = OrderedDict([(imtx, {}) for imtx in self.imts])
            for imtx in self.imts:
                for res_type in self.types[gmpe][imtx]:
                    gsim = GSIM_LIST[gmpe]()
                    mean, stddev = gsim.get_mean_and_stddevs(
                        context["Sites"],
                        context["Rupture"],
                        context["Distances"],
                        imt.from_string(imtx),
                        [res_type])
                    expected[gmpe][imtx]["Mean"] = mean
                    expected[gmpe][imtx][res_type] = stddev[0]
        context["Expected"] = expected
        return context
                    
    def calculate_residuals(self, context):
        """
        Calculate the residual terms
        """
        # Calculate residual
        residual = {}
        for gmpe in self.gmpe_list:
            residual[gmpe] = {}
            for imtx in self.imts:
                residual[gmpe][imtx] = {}
                obs = np.log(context["Observations"][imtx])
                mean = context["Expected"][gmpe][imtx]["Mean"]
                total_stddev = context["Expected"][gmpe][imtx]["Total"]
                residual[gmpe][imtx]["Total"] = (obs - mean) / total_stddev
                if "Inter event" in self.residuals[gmpe][imtx].keys():
                    inter, intra = self._get_random_effects_residuals(
                        obs,
                        mean,
                        context["Expected"][gmpe][imtx]["Inter event"],
                        context["Expected"][gmpe][imtx]["Intra event"])
                    residual[gmpe][imtx]["Inter event"] = inter
                    residual[gmpe][imtx]["Intra event"] = intra
        context["Residual"] = residual
        return context

    def _get_random_effects_residuals(self, obs, mean, inter, intra):
        """
        Calculates the random effects residuals using the inter-event
        residual formula described in Abrahamson & Youngs (1992) Eq. 10
        """
        nvals = float(len(mean))
        inter_res = ((inter ** 2.) * sum(obs - mean)) /\
                     (nvals * (inter ** 2.) + (intra ** 2.))
        intra_res = obs - (mean + inter_res)
        return inter_res / inter, intra_res / intra

    def get_residual_statistics(self):
        """
        Retreives the mean and standard deviation values of the residuals
        """
        statistics = OrderedDict([(gmpe, {}) for gmpe in self.gmpe_list])
        for gmpe in self.gmpe_list:
            for imtx in self.imts:
                statistics[gmpe][imtx] = {}
                for res_type in self.types[gmpe][imtx]:
                    data = {
                        "Mean": np.mean(
                            self.residuals[gmpe][imtx][res_type]),
                        "Std Dev": np.std(
                            self.residuals[gmpe][imtx][res_type])}
                    statistics[gmpe][imtx][res_type] = data
        return statistics

class Likelihood(Residuals):
    """
    Implements the likelihood function of Scherbaum et al. (2004)
    """
        
    def get_likelihood_values(self):
        """
        Returns the likelihood values for Total, plus inter- and intra-event
        residuals according to Equation 9 of Scherbaum et al (2004)
        """
        statistics = self.get_residual_statistics()
        lh_values = OrderedDict([(gmpe, {}) for gmpe in self.gmpe_list])
        for gmpe in self.gmpe_list:
            for imtx in self.imts:
                lh_values[gmpe][imtx] = {}
                for res_type in self.types[gmpe][imtx]:
                    zvals = np.fabs(self.residuals[gmpe][imtx][res_type])
                    l_h = erf(zvals / sqrt(2.))
                    lh_values[gmpe][imtx][res_type] = l_h
                    statistics[gmpe][imtx][res_type]["Median LH"] =\
                        scoreatpercentile(l_h, 50.0)
        return lh_values, statistics


class LLH(Residuals):
    """
    Implements of average sample log-likelihood estimator from
    Scherbaum et al (2009)
    """
    def get_loglikelihood_values(self):
        log_residuals = OrderedDict([(gmpe, np.array([]))
                                      for gmpe in self.gmpe_list])
        llh = OrderedDict([(gmpe, None) for gmpe in self.gmpe_list])
        for gmpe in self.gmpe_list:
            for imtx in self.imts:
                asll = np.log2(norm.pdf(self.residuals[gmpe][imtx]["Total"], 
                               0., 
                               1.0))
                log_residuals[gmpe] = np.hstack([log_residuals[gmpe], asll])
            llh[gmpe] = (1. / float(len(log_residuals[gmpe]))) *\
                np.sum(log_residuals[gmpe])
        # Get weights
        weights = np.array([2.0 ** llh[gmpe] for gmpe in self.gmpe_list])
        weights = weights / np.sum(weights)
        model_weights = OrderedDict([
            (gmpe, weights[iloc]) for iloc, gmpe in enumerates(self.gmpe_list)]
            )
        return llh, model_weights

class EDR(Residuals):
    """
    Implements the Euclidean Distance-Based Ranking Method for GMPE selection
    by Kale & Akkar (2013)
    Kale, O., and Akkar, S. (2013) A New Procedure for Selecting and Ranking
    Ground Motion Predicion Equations (GMPEs): The Euclidean Distance-Based
    Ranking Method
    """
    def get_edr_values(self, bandwidth=0.01, multiplier=3.0):
        """
        Calculates the EDR values for each GMPE
        :param float bandwidth:
            Discretisation width
        :param float multiplier:
            "Multiplier of standard deviation (equation 8 of Kale and Akkar)
        """
        edr_values = OrderedDict([(gmpe, {}) for gmpe in self.gmpe_list])
        for gmpe in gmpe_list:
            obs, expected, stddev = self._get_gmpe_information(gmpe)
            results = self._get_edr(obs,
                                    expected,
                                    stddev,
                                    bandwidth,
                                    multiplier)
            edr_values["MDE Norm"] = results[0]
            edr_values["sqrt Kappa"] = results[1]
            edr_values["EDR"] = results[2]
        return edr_values

    def _get_gmpe_information(self, gmpe):
        """
        Extract the observed ground motions, expected and total standard
        deviation for the GMPE (aggregating over all IMS)
        """
        obs = np.array([], dtype=float)
        expected = np.array([], dtype=float)
        stddev = np.array([], dtype=float)
        for imtx in self.imts:
            for context in self.contexts:
                obs = np.hstack([obs, np.log(context["Observations"][imtx])])
                expected = np.hstack([expected,
                                      context["Expected"][gmpe][imtx]["Mean"]])
                stddev = np.hstack([stddev,
                                    context["Expected"][gmpe][imtx]["Total"]])
        return obs, expected, stddev

    def _get_edr(self, obs, expected, stddev, bandwidth=0.01, multiplier=3.0):
        """
        Calculated the Euclidean Distanced-Based Rank for a set of
        observed and expected values from a particular GMPE
        """
        nvals = len(obs)
        min_d = bandwidth / 2.
        kappa = self._get_kappa(obs, expected)
        mu_d = obs - expected
        d1c = np.fabs(obs - (expected - (multiplier * stddev)))
        d2c = np.fabs(obs - (expected + (multiplier * stddev)))
        dc_max = ceil(np.max(np.array([np.max(d1c), np.max(d2c)])))
        num_d = len(np.arange(min_d, dc_max, bandwidth))
        mde = np.zeros(nvals)
        for iloc in range(0, num_d):
            d_val = (dmin + (iloc * dd)) * np.ones(nvals)
            d_1 = (dmin + (iloc * dd)) * np.ones(nvals)
            d_2 = (dmin + (iloc * dd)) * np.ones(nvals)
            p_1 = norm.cdf((d1 - mu_d) / stddev) -\
                norm.cdf((-d1 - mu_d) / stddev)
            p_2 = norm.cdf((d2 - mu_d) / stddev) -\
                norm.cdf((-d2 - mu_d) / stddev)
            mde += (p_2 - p_1) * dval
        inv_n = 1.0 / float(nvals)
        mde_norm = np.sqrt(inv_n * np.sum(mde ** 2.))
        edr = np.sqrt(kappa * inv_n * np.sum(mde ** 2.))
        return mde_norm, np.sqrt(kappa), edr


    def _get_kappa(self, obs, expected):
        """
        Returns the correction factor kappa
        """
        mu_a = np.mean(obs)
        mu_y = np.mean(expected)
        b_1 = np.sum((obs - mu_a) * (expected - mu_y)) /\
            np.sum((obs - mu_a) ** 2.)
        b_0 = mu_y - b_1 * mu_a
        y_c =  expected - ((b_0 + b_1 * obs) - obs)
        de_orig = np.sum((obs - expected) ** 2.)
        de_corr = np.sum((obs - y_c) ** 2.)
        return de_orig / de_corr
        
        



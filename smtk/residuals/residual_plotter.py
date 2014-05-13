#!/usr/bin/env/python 

"""
Class to hold GMPE residual plotting functions
"""

import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil
from scipy.stats import norm, linregress
from smtk.residuals.gmpe_residuals import Residuals, Likelihood

def _save_image(filename, filetype='png', resolution=300):
    """
    If filename is specified, saves the image
    :param str filename:
        Name of the file
    :param str filetype:
        Type of file
    :param int resolution:
        DPI resolution of the output figure
    """
    if filename:
        filename, filetype, resolution = build_filename(filename,
                                                        filetype,
                                                        resolution)
        plt.savefig(filename, dpi=resolution, format=filetype)
    else:
        pass
    return


class ResidualPlot(object):
    """
    Class to create a simple histrogram of strong ground motion residuals 
    """
    def __init__(self, residuals, gmpe, imt, filename=None, filetype="png",
        dpi=300, **kwargs):
        """
        :param residuals:
            Residuals as instance of :class: smtk.gmpe_residuals.Residuals
        :param str gmpe:
            Choice of GMPE
        :param str imt:
            Choice of IMT
        """
        kwargs.setdefault('plot_type', "log")
        kwargs.setdefault('distance_type', "rjb")
        kwargs.setdefault("figure_size", (7, 5))
        self._assertion_check(residuals)
        self.residuals = residuals
        if not gmpe in residuals.gmpe_list:
            raise ValueError("No residual data found for GMPE %s" % gmpe)
        if not imt in residuals.imts:
            raise ValueError("No residual data found for IMT %s" % imt)
        self.gmpe = gmpe
        self.imt = imt
        self.filename = filename
        self.filetype = filetype
        self.dpi = dpi
        self.num_plots = len(residuals.types[gmpe][imt])
        self.distance_type = kwargs["distance_type"]
        self.plot_type = kwargs["plot_type"]
        self.figure_size = kwargs["figure_size"]
        self.create_plot()

    def _assertion_check(self, residuals):
        """
        Checks that residuals is an instance of the residuals class
        """
        assert isinstance(residuals, Residuals)

    def create_plot(self, bin_width=0.5):
        """
        Creates a histogram plot
        """
        data = self.residuals.residuals[self.gmpe][self.imt]
        statistics = self.residuals.get_residual_statistics()
        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout(True)
        if self.num_plots > 1:
            nrow = 2
            ncol = 2
        else:
            nrow = 1
            ncol = 1
        tloc = 1
        for res_type in data.keys():
            self._density_plot(
                plt.subplot(nrow, ncol, tloc),
                data,
                res_type,
                statistics,
                bin_width)
            tloc += 1
        plt.show()
        _save_image(self.filename, self.dpi, self.filetype)
           
    def _density_plot(self, ax, data, res_type, statistics, bin_width=0.5):
        """
        Plots the density distribution on the subplot axis
        """
        vals, bins = self.get_histogram_data(data[res_type], bin_width)
        ax.bar(bins[:-1], vals, width=0.95 * bin_width, color="LightSteelBlue",
               edgecolor="k")
        # Get equivalent normal distribution
        mean = statistics[self.gmpe][self.imt][res_type]["Mean"]
        stddev = statistics[self.gmpe][self.imt][res_type]["Std Dev"]
        #print mean, stddev
        xdata = np.arange(bins[0], bins[-1] + 0.01, 0.01)
        ax.plot(xdata, norm.pdf(xdata, mean, stddev), '-',
                color="LightSlateGrey", linewidth=2.0)
        ax.plot(xdata, norm.pdf(xdata, 0.0, 1.0), 'k-', linewidth=2.0)
        ax.set_xlabel("Z (%s)" % self.imt, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        title_string = "%s - %s\n Mean = %7.3f, Std Dev = %7.3f" %(self.gmpe,
                                                                   res_type,
                                                                   mean,
                                                                   stddev)
        ax.set_title(title_string, fontsize=12)
        
    def get_histogram_data(self, data, bin_width=0.5):
        """
        Retreives the histogram of the residuals
        """
        bins = np.arange(np.floor(np.min(data)),
                         np.ceil(np.max(data)) + bin_width,
                         bin_width)
        vals = np.histogram(data, bins, density=True)[0]
        return vals.astype(float), bins
        

class LikelihoodPlot(ResidualPlot):
    """

    """

    def _assertion_check(self, residuals):
        """

        """
        assert isinstance(residuals, Likelihood)

    def create_plot(self, bin_width=0.1):
        """
        Creates a histogram plot
        """
        #data = self.residuals.residuals[self.gmpe][self.imt]
        lh_vals, statistics = self.residuals.get_likelihood_values()
        lh_vals = lh_vals[self.gmpe][self.imt]
        statistics = statistics[self.gmpe][self.imt]
        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout(True)
        if self.num_plots > 1:
            nrow = 2
            ncol = 2
        else:
            nrow = 1
            ncol = 1
        tloc = 1
        for res_type in lh_vals.keys():
            self._density_plot(
                plt.subplot(nrow, ncol, tloc),
                lh_vals[res_type],
                res_type,
                statistics[res_type],
                bin_width)
            tloc += 1
        plt.show()
        _save_image(self.filename, self.dpi, self.filetype)
        

    def _density_plot(self, ax, lh_values, res_type, statistics, 
            bin_width=0.1):
        """
        """
        vals, bins = self.get_histogram_data(lh_values, bin_width)
        ax.bar(bins[:-1], vals, width=0.95 * bin_width, color="LightSteelBlue",
               edgecolor="k")
        # Get equivalent normal distribution
        median_lh = statistics["Median LH"]
        ax.set_xlabel("LH (%s)" % self.imt, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_xlim(0., 1.0)
        title_string = "%s - %s\n Median LH = %7.3f" %(self.gmpe,
                                                       res_type,
                                                       median_lh)
        ax.set_title(title_string, fontsize=12)
    
    
    def get_histogram_data(self, lh_values, bin_width=0.1):
        """

        """
        bins = np.arange(0.0, 1.0 + bin_width, bin_width)
        vals = np.histogram(lh_values, bins, density=True)[0]
        return vals.astype(float), bins


class ResidualWithDistance(ResidualPlot):
    """

    """
    def __init__(self, residuals, gmpe, imt, filename=None, filetype="png", 
         dpi=300, **kwargs):
         """

         """
         super(ResidualWithDistance, self).__init__(residuals, gmpe, imt,
             filename, filetype, dpi, **kwargs)

    def create_plot(self):
        """

        """
        data = self.residuals.residuals[self.gmpe][self.imt]
        distances = self._get_distances()
        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout(True)
        if self.num_plots > 1:
            nrow = 3
            ncol = 1
        else:
            nrow = 1
            ncol = 1
        tloc = 1
        for res_type in data.keys():
            self._residual_plot(
                plt.subplot(nrow, ncol, tloc),
                distances,
                data,
                res_type)
            tloc += 1
        plt.show()
        _save_image(self.filename, self.dpi, self.filetype)


    def _residual_plot(self, ax, distances, data, res_type):
        """

        """
        slope, intercept, _, pval, _ = linregress(distances, data[res_type])
        model_x = np.arange(np.min(distances),
                            np.max(distances) + 1.0,
                            1.0)
        model_y = intercept + slope * model_x
        if self.plot_type == "log":
            ax.semilogx(distances,
                        data[res_type],
                        'o',
                        markeredgecolor='Gray',
                        markerfacecolor='LightSteelBlue')
            ax.semilogx(model_x, model_y, 'r-', linewidth=2.0)
            ax.set_xlim(0.1, 10.0 ** (ceil(np.log10(np.max(distances)))))
        else:
            ax.plot(distances,
                    data[res_type],
                    'o',
                    markeredgecolor='Gray',
                    markerfacecolor='LightSteelBlue')
            ax.plot(model_x, model_y, 'r-', linewidth=2.0)
            ax.set_xlim(0, np.max(distances))
        max_lim = ceil(np.max(np.fabs(data[res_type])))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_xlabel("%s Distance (km)" % self.distance_type, fontsize=12)
        ax.set_ylabel("Z (%s)" % self.imt, fontsize=12)
        #title_string = "%s - %s (p = %.5e)" %(self.gmpe, res_type, pval)
        title_string = "%s - %s\n Slope = %.4e, Intercept = %7.3f"\
                       " p = %.6e " % (self.gmpe, res_type, slope, intercept,
                                       pval)
        ax.set_title(title_string, fontsize=12)

    def _get_distances(self):
        """

        """
        distances = np.array([])
        for ctxt in self.residuals.contexts:
            distances = np.hstack([
                distances,
                getattr(ctxt["Distances"], self.distance_type)])
        return distances


class ResidualWithMagnitude(ResidualPlot):
    """

    """
    def __init__(self, residuals, gmpe, imt, filename=None, filetype="png", 
         dpi=300, **kwargs):
         """

         """
         super(ResidualWithMagnitude, self).__init__(residuals, gmpe, imt,
             filename, filetype, dpi, **kwargs)

    def create_plot(self):
        """
        Creates the plot
        """
        data = self.residuals.residuals[self.gmpe][self.imt]
        magnitudes = self._get_magnitudes()
        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout(True)
        if self.num_plots > 1:
            nrow = 3
            ncol = 1
        else:
            nrow = 1
            ncol = 1
        tloc = 1
        for res_type in data.keys():
            self._residual_plot(
                plt.subplot(nrow, ncol, tloc),
                magnitudes,
                data,
                res_type)
            tloc += 1
        plt.show()
        _save_image(self.filename, self.dpi, self.filetype)


    def _residual_plot(self, ax, magnitudes, data, res_type):
        """
        Plots the residuals with magnitude
        """
        slope, intercept, _, pval, _ = linregress(magnitudes, data[res_type])
        model_x = np.arange(np.min(magnitudes),
                            np.max(magnitudes) + 1.0,
                            1.0)
        model_y = intercept + slope * model_x
        ax.plot(magnitudes,
                data[res_type],
                'o',
                markeredgecolor='Gray',
                markerfacecolor='LightSteelBlue')
        ax.plot(model_x, model_y, 'r-', linewidth=2.0)
        ax.set_xlim(floor(np.min(magnitudes)), ceil(np.max(magnitudes)))
        max_lim = ceil(np.max(np.fabs(data[res_type])))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_xlabel("Magnitude", fontsize=12)
        ax.set_ylabel("Z (%s)" % self.imt, fontsize=12)
        #title_string = "%s - %s (p = %.5e)" %(self.gmpe, res_type, pval)
        title_string = "%s - %s\n Slope = %.4e, Intercept = %7.3f"\
                       " p = %.6e " % (self.gmpe, res_type, slope, intercept,
                                       pval)
        ax.set_title(title_string, fontsize=12)

    def _get_magnitudes(self):
        """
        Returns an array of magnitudes equal in length to the number of
        residuals
        """
        magnitudes = np.array([])
        for ctxt in self.residuals.contexts:
            magnitudes = np.hstack([
                magnitudes,
                ctxt["Rupture"].mag * np.ones(len(ctxt["Distances"].repi))])
        return magnitudes
    

class ResidualWithVs30(ResidualPlot):
    """

    """
    def __init__(self, residuals, gmpe, imt, filename=None, filetype="png", 
         dpi=300, **kwargs):
         """

         """
         super(ResidualWithVs30, self).__init__(residuals, gmpe, imt,
             filename, filetype, dpi, **kwargs)

    def create_plot(self):
        """

        """
        data = self.residuals.residuals[self.gmpe][self.imt]
        vs30 = self._get_vs30()
        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout(True)
        if self.num_plots > 1:
            nrow = 3
            ncol = 1
        else:
            nrow = 1
            ncol = 1
        tloc = 1
        for res_type in data.keys():
            self._residual_plot(
                plt.subplot(nrow, ncol, tloc),
                vs30,
                data,
                res_type)
            tloc += 1
        plt.show()
        _save_image(self.filename, self.dpi, self.filetype)


    def _residual_plot(self, ax, vs30, data, res_type):
        """

        """
        slope, intercept, _, pval, _ = linregress(vs30, data[res_type])
        model_x = np.arange(np.min(vs30),
                            np.max(vs30) + 1.0,
                            1.0)
        model_y = intercept + slope * model_x
        ax.plot(vs30,
                data[res_type],
                'o',
                markeredgecolor='Gray',
                markerfacecolor='LightSteelBlue')
        ax.plot(model_x, model_y, 'r-', linewidth=2.0)
        ax.set_xlim(0.1, np.max(vs30))
        max_lim = ceil(np.max(np.fabs(data[res_type])))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_xlabel("Vs30 (m/s)", fontsize=12)
        ax.set_ylabel("Z (%s)" % self.imt, fontsize=12)
        title_string = "%s - %s\n Slope = %.4e, Intercept = %7.3f"\
                       " p = %.6e " % (self.gmpe, res_type, slope, intercept,
                                       pval)
        ax.set_title(title_string, fontsize=12)

    def _get_vs30(self):
        """

        """
        vs30 = np.array([])
        for ctxt in self.residuals.contexts:
            vs30 = np.hstack([vs30, ctxt["Sites"].vs30])
        return vs30

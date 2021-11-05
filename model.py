from scipy.interpolate import interp1d
import numpy as np
import pycmpfit
# phase, mean, _, _, _, _, = np.loadtxt(
#     '/Users/jiawenyang/Packages/LCfitter/LCfPCA_He/bandSpecific_B.txt',
#  skiprows=1, unpack=True)  # FIXME


def stretch_model_val(t, template_phase, template_mean, theta,
                      fit_range=[-10, 50]):
    """Calculate stretch model value given time t.
    See https://arxiv.org/pdf/1107.2404.pdf

    Args:
        t (float or array_like): time to calculate model value.
        template_phase (array_like): phases of template used to fit
         the light curve.
        template_mean (array_like): magnitudes of template used to
         fit the light curve.
        theta (array_like): array_like of length 4,
         [max mag, max time, s1, s2].
        fit_range (array_like, optional): array_like of length 2,
         defines the range of phase needs to be fit. Defaults to [-10,50]
    Returns:
        float or array_like: magnitudes of model at give t.
    """
    maxm, maxt, s1, s2 = theta
    imod0 = interp1d(template_phase, template_mean+maxm,
                     fill_value='extrapolate')

    return np.where(t < maxt,
                    np.where((t-maxt)/s1 <
                             fit_range[0], np.nan, imod0((t-maxt)/s1)),
                    np.where((t-maxt)/s2 > fit_range[1],
                             np.nan, imod0((t-maxt)/s2)))

    # if t < maxt:
    #     if (t-maxt)/s1 < -10:
    #         return np.nan
    #     else:
    #         return imod0((t-maxt)/s1)
    # else:
    #     if (t-maxt)/s2 > 50:
    #         return np.nan
    #     else:
    #         return imod0((t-maxt)/s2)


def stretch_userfunc(m, n, theta, private_data):
    """User function to be optimized.

    Args:
        m (int): Length of data.
        n (int): Length of parameters (theta).
        theta (array-like): Parameters.
        private_data (dict): Data to be fit.

    Returns:
        dict: Deviation of model from data.
    """
    tt, mm, emm = [private_data[ii] for ii in ['x', 'y', 'ey']]
    template_days, template_magnitudes = [
        private_data[ii] for ii in ['template_days', 'template_magnitudes']]
    fit_range = private_data['fit_range']
    devs = np.zeros((m), dtype=np.float64)

    y_model = stretch_model_val(
        tt, template_days, template_magnitudes, theta, fit_range=fit_range)

    devs = np.where(np.isnan(y_model), 0, (mm-y_model)/emm)
    user_dict = {"deviates": devs}
    return user_dict


def stretch_fit(days, magnitudes, emagnitudes,
                template_days, template_magnitudes,
                theta=None, fit_range=[-10, 50], bounds=None, fixed=None):
    """Fit light curve use two streches.

    Args:
        days (array-like): days of light curve to be fit.
        magnitudes (arra-like): magnitudes of light curve to be fit.
        emagnitudes (array-like): errors of magnitudes.
        template_days (array-like): days of template light curve.
        template_magnitudes (array-like): magnitudes of template lighht curve,
        theta (array-like, optional): Initial parameters value.
         Defaults to None.
        fit_range (array_like, optional): array_like of length 2,
         defines the range of phase needs to be fit. Defaults to [-10,50]
        bounds (2 tuple of array-like, optional): gives the lower an upper
         boundary of parameters. Defaults to None.
        fixed (array-like of boolean value): if True, fix that
         parameter to inital value. Defaults to None.
    """
    max_mag = min(magnitudes)
    max_date = days[np.argmin(magnitudes)]
    if theta is None:
        theta = np.array([max_mag, max_date, 1, 1])

    py_mp_par = list(pycmpfit.MpPar() for i in range(len(theta)))
    if bounds is not None:
        for ii in range(len(theta)):
            py_mp_par[ii].limited = [1, 1]
            py_mp_par[ii].limits = [bounds[0][ii], bounds[1][ii]]
    if fixed is not None:
        for ii in range(len(fixed)):
            py_mp_par[ii].fixed = fixed[ii]
    user_data = {'x': days, 'y': magnitudes, 'ey': emagnitudes,
                 'template_days': template_days,
                 'template_magnitudes': template_magnitudes,
                 'fit_range': fit_range}
    fit = pycmpfit.Mpfit(stretch_userfunc, len(days), theta,
                         private_data=user_data, py_mp_par=py_mp_par)

    fit.mpfit()

    return theta, fit.result

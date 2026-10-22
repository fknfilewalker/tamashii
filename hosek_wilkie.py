"""Hosek-Wilkie sky [2012] and solar radiance [2013] models, ported from
ArHosekSkyModel.c 1.4a (https://cgg.mff.cuni.cz/projects/SkylightModelling/).

Turbidity, albedo and solar elevation are cooked into nine parameters per
wavelength here, so the shader only evaluates the formula. Angles are Z-up with
the observer on +Z. Defined only above the horizon.
"""

import numpy as np

from .hosek_wilkie_tables import (
    LIMB_DARKENING,
    SKY_PARAMS,
    SKY_RADIANCE,
    SOLAR_RADIUS,
    SUN_PARAMS,
    WAVELENGTHS,
    WAVELENGTH_STEP,
)
from .spectra import (
    SPECTRAL_RESPONSE,
    SPECTRAL_RESPONSE_START,
    SPECTRAL_RESPONSE_STEP,
    XYZ_TO_SRGB,
)

TURBIDITY_MIN, TURBIDITY_MAX = 1.0, 10.0
EARTH_MEAN_RADIUS = 6371.01      # km
ASTRONOMICAL_UNIT = 149597890.0  # km
SUN_SEGMENTS = 45

def rgb_weights() -> np.ndarray:
    """(11, 3) matrix taking spectral radiance to linear sRGB.

    The wavelengths are point samples, so each stands for a WAVELENGTH_STEP bin.
    """
    xyz = np.zeros((len(WAVELENGTHS), 3))
    for channel, wavelength in enumerate(WAVELENGTHS):
        index = int((wavelength - SPECTRAL_RESPONSE_START) / SPECTRAL_RESPONSE_STEP)
        if 0 <= index < len(SPECTRAL_RESPONSE):
            xyz[channel] = SPECTRAL_RESPONSE[index]
    return (xyz * WAVELENGTH_STEP) @ XYZ_TO_SRGB.T


def _quintic_bezier(t: float) -> np.ndarray:
    """Bernstein basis of degree five: the model interpolates over elevation with it."""
    s = 1.0 - t
    return np.array([s**5, 5 * s**4 * t, 10 * s**3 * t**2, 10 * s**2 * t**3, 5 * s * t**4, t**5])


def _config_weights(turbidity: float, albedo: float):
    """The four (albedo, turbidity) corners and their weights.

    The reference floors turbidity and skips the upper level at turbidity 10.
    """
    turbidity = float(np.clip(turbidity, TURBIDITY_MIN, TURBIDITY_MAX))
    albedo = float(np.clip(albedo, 0.0, 1.0))
    low = int(turbidity)
    rem = turbidity - low
    corners = [(0, low - 1, (1.0 - albedo) * (1.0 - rem)), (1, low - 1, albedo * (1.0 - rem))]
    if low != 10:
        corners += [(0, low, (1.0 - albedo) * rem), (1, low, albedo * rem)]
    return corners


def cook(turbidity: float, albedo: float, elevation_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the dataset to one sky: (11, 9) parameters and (11,) radiance scale."""
    elevation = float(np.clip(elevation_rad, 0.0, 0.5 * np.pi))
    basis = _quintic_bezier((elevation / (0.5 * np.pi)) ** (1.0 / 3.0))

    params = np.zeros((len(WAVELENGTHS), 9))
    radiance = np.zeros(len(WAVELENGTHS))
    for albedo_index, turbidity_index, weight in _config_weights(turbidity, albedo):
        if weight == 0.0:
            continue
        # contracted against the six control points
        params += weight * np.einsum("k,wkp->wp", basis, SKY_PARAMS[:, albedo_index, turbidity_index])
        radiance += weight * (SKY_RADIANCE[:, albedo_index, turbidity_index] @ basis)
    return params, radiance


def sky_radiance(params: np.ndarray, radiance: np.ndarray, theta: float, gamma: float) -> np.ndarray:
    """Spectral sky radiance for cooked parameters, one value per wavelength."""
    cos_theta = np.cos(theta)
    if cos_theta < 0.0:
        return np.zeros(len(WAVELENGTHS))
    cos_gamma = np.cos(gamma)
    a, b, c, d, e, f, g, h, i = (params[:, k] for k in range(9))
    mie = (1.0 + cos_gamma**2) / np.power(1.0 + i * i - 2.0 * i * cos_gamma, 1.5)
    return (
        (1.0 + a * np.exp(b / (cos_theta + 0.01)))
        * (c + d * np.exp(e * gamma) + f * cos_gamma**2 + g * mie + h * np.sqrt(cos_theta))
        * radiance
    )


def sun_radiance(turbidity: float, elevation_rad: float) -> np.ndarray:
    """Direct radiance at the disc centre per wavelength, before limb darkening.

    Depends only on the sun's own elevation, so it is constant across the disc.
    """
    elevation = float(np.clip(elevation_rad, 0.0, 0.5 * np.pi))
    turbidity = float(np.clip(turbidity, TURBIDITY_MIN, TURBIDITY_MAX))

    # the reference clamps the top level down and leans fully on it
    low = int(turbidity) - 1
    frac = turbidity - (low + 1)
    if low == 9:
        low, frac = 8, 1.0

    segment = min(int((2.0 * elevation / np.pi) ** (1.0 / 3.0) * SUN_SEGMENTS), SUN_SEGMENTS - 1)
    x = elevation - (segment / SUN_SEGMENTS) ** 3 * (0.5 * np.pi)

    def at(turbidity_index: int) -> np.ndarray:
        # four coefficients per segment, descending power order
        coefs = SUN_PARAMS[:, turbidity_index, segment]
        return coefs[:, 3] + x * (coefs[:, 2] + x * (coefs[:, 1] + x * coefs[:, 0]))

    low_value = at(low)
    return low_value if frac == 0.0 else low_value + (at(low + 1) - low_value) * frac


def limb_darkening_rgb(turbidity: float, elevation_rad: float) -> np.ndarray:
    """(6, 3) sRGB coefficients of the disc's limb-darkening polynomial.

    Spectral->sRGB is linear, so it commutes with the polynomial and the two
    collapse into one -- the shader never sees a spectrum for the sun.
    """
    return (LIMB_DARKENING * sun_radiance(turbidity, elevation_rad)[:, None]).T @ rgb_weights()


def pack(turbidity: float, albedo: float, elevation_rad: float, azimuth_rad: float) -> tuple[np.ndarray, dict]:
    """Flatten what `emitters::HosekWilkie` reads into one float32 buffer.

    Header (sun direction, 1/sin^2 of the disc radius), sky parameters, radiance
    scale, channel-to-sRGB matrix, limb-darkening polynomial.
    """
    params, radiance = cook(turbidity, albedo, elevation_rad)

    sun = np.array([
        np.cos(azimuth_rad) * np.cos(elevation_rad),
        np.sin(azimuth_rad) * np.cos(elevation_rad),
        np.sin(elevation_rad),
    ])
    header = np.concatenate([sun, [1.0 / np.sin(SOLAR_RADIUS) ** 2], np.zeros(4)])

    data = np.concatenate([
        header,
        params.ravel(),
        radiance,
        rgb_weights().ravel(),
        limb_darkening_rgb(turbidity, elevation_rad).ravel(),
    ]).astype(np.float32)

    return data, {"channels": len(WAVELENGTHS)}


# -- solar position ---------------------------------------------------------


def _trunc_div(a, b):
    """Truncating integer division, as C does -- the Julian day formula needs it."""
    return np.trunc(np.asarray(a, dtype=np.float64) / b)


def sun_position(latitude: float, longitude: float, timezone: float,
                 year, month, day, hour, minute=0.0, second=0.0):
    """Solar elevation and azimuth in radians, by Blanco-Muriel et al. (2001).

    Degrees north/east, `timezone` offset from UTC, `hour` local clock time.
    Date and time may be arrays. Azimuth zero puts the sun on +X.
    """
    year, month, day = np.asarray(year), np.asarray(month), np.asarray(day)
    hour, minute, second = np.asarray(hour, dtype=np.float64), np.asarray(minute), np.asarray(second)
    scalar = np.ndim(year) == 0 and np.ndim(hour) == 0

    decimal_hours = hour - timezone + (minute + second / 60.0) / 60.0

    aux = _trunc_div(month - 14, 12)
    julian_day = (
        _trunc_div(1461 * (year + 4800 + aux), 4)
        + _trunc_div(367 * (month - 2 - 12 * aux), 12)
        - _trunc_div(3 * _trunc_div(year + 4900 + aux, 100), 4)
        + day
        - 32075
    )
    # days since JD 2451545.0 (noon, 1 Jan 2000 UT)
    elapsed = (julian_day - 0.5 + decimal_hours / 24.0) - 2451545.0

    # ecliptic
    omega = 2.1429 - 0.0010394594 * elapsed
    mean_longitude = 4.8950630 + 0.017202791698 * elapsed
    anomaly = 6.2400600 + 0.0172019699 * elapsed
    ecliptic_longitude = (mean_longitude + 0.03341607 * np.sin(anomaly)
                          + 0.00034894 * np.sin(2 * anomaly) - 0.0001134
                          - 0.0000203 * np.sin(omega))
    ecliptic_obliquity = 0.4090928 - 6.2140e-9 * elapsed + 0.0000396 * np.cos(omega)

    # celestial
    sin_ecliptic_longitude = np.sin(ecliptic_longitude)
    right_ascension = np.arctan2(np.cos(ecliptic_obliquity) * sin_ecliptic_longitude,
                                 np.cos(ecliptic_longitude))
    right_ascension = np.where(right_ascension < 0.0, right_ascension + 2 * np.pi, right_ascension)
    declination = np.arcsin(np.sin(ecliptic_obliquity) * sin_ecliptic_longitude)

    # local horizon
    greenwich_sidereal_time = 6.6974243242 + 0.0657098283 * elapsed + decimal_hours
    local_sidereal_time = np.radians(greenwich_sidereal_time * 15 + longitude)
    latitude_rad = np.radians(latitude)
    cos_latitude, sin_latitude = np.cos(latitude_rad), np.sin(latitude_rad)

    hour_angle = local_sidereal_time - right_ascension
    cos_hour_angle = np.cos(hour_angle)

    zenith = np.arccos(np.clip(cos_latitude * cos_hour_angle * np.cos(declination)
                               + np.sin(declination) * sin_latitude, -1.0, 1.0))
    azimuth = np.arctan2(-np.sin(hour_angle),
                         np.tan(declination) * cos_latitude - sin_latitude * cos_hour_angle)
    azimuth = np.where(azimuth < 0.0, azimuth + 2 * np.pi, azimuth)

    # geocentric -> topocentric parallax
    zenith = zenith + (EARTH_MEAN_RADIUS / ASTRONOMICAL_UNIT) * np.sin(zenith)

    elevation, azimuth = 0.5 * np.pi - zenith, azimuth - np.pi
    return (float(elevation), float(azimuth)) if scalar else (elevation, azimuth)

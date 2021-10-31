# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from collections import defaultdict
from itertools import count
from astropy.table import Table
from astropy.io import fits

from chandra_aca.aca_image import ACAImage
from mica.archive import aca_dark
from chandra_aca.aca_image import AcaPsfLibrary


# Lists of row, col indices for the edge of an 8x8 image
# plus the corner pixels
EDGE_ROWS_8x8 = [0, 0, 0, 0, 0, 0, 0, 0,
                 7, 7, 7, 7, 7, 7, 7, 7,
                 1, 2, 3, 4, 5, 6,
                 1, 2, 3, 4, 5, 6,
                 1, 1, 6, 6]
EDGE_COLS_8x8 = [0, 1, 2, 3, 4, 5, 6, 7,
                 0, 1, 2, 3, 4, 5, 6, 7,
                 0, 0, 0, 0, 0, 0,
                 7, 7, 7, 7, 7, 7,
                 1, 6, 1, 6]

APL = AcaPsfLibrary()


def get_pix_estimates(pix_vals, pix_times, times, image_readout_period=4.1, seed=0):
    """
    Using the time history for each pixel observed as an "edge" pixel, calculate
    an estimate of that pixel at every image time.

    This uses a time-window median filter to filter outliers and then a linear
    interpolation.

    :param pix_vals: observed values of the pixel
    :param pix_times: times of observed values
    :param times: all image times (times for which a pixel value is desired)
    :param image_readout_period: image readout period (seconds).  used to determine window.
    :returns: np.array of estimated pixel values at `times`
    """
    # Add five "seed" samples at times spaces by image_readout_period before and
    # after the real data.  This avoids less controlled edge effects.
    pre_times = times[0] + (np.array([-5, -4, -3, -2, -1]) * image_readout_period)
    post_times = times[-1] + (np.array([1, 2, 3, 4, 5]) * image_readout_period)
    times_pix = np.hstack([pre_times, np.array(pix_times), post_times])
    vals_pix = np.hstack([np.repeat(seed, 5), np.array(pix_vals), np.repeat(seed, 5)])

    # Loop over the observed pixel values and apply time-window median filter.
    times_filtered = []
    values_filtered = []
    for i in range(2, len(times_pix) - 2):

        # This time-window median confirms that the 5 (2 before 2 after) samples closest
        # to i are within 5.5 of the image_readout_periods / sampling interval and skips
        # this i if there aren't enough samples.
        if (times_pix[i + 2] - times_pix[i - 2]) < (5.5 * image_readout_period):
            times_filtered.append(times_pix[i])
            values_filtered.append(np.median(vals_pix[i-2:i+2]))

    # Linearly interpolate the filtered data
    return np.interp(x=times, xp=times_filtered, fp=values_filtered)


def get_estimated_backgrounds(imgs, ccd_bgd=None):
    """
    Get estimated / dynamic background for images.

    This uses the imgs to determine background estimates using the edge algorithm.

    :param imgs: list of ACAImages with raw ACA images (DN)
    :param ccd_bgd: dark cal image used for seed values for algorithm
    :return: list of ACAImages of len(imgs) with images of estimated background
    """
    dark = defaultdict(list)
    dark_times = defaultdict(list)
    all_times = []
    for i, img in enumerate(imgs):
        all_times.append(img.meta['TIME'])
        for r, c in zip(EDGE_ROWS_8x8, EDGE_COLS_8x8):
            row = img.meta['IMGROW0'] + r
            col = img.meta['IMGCOL0'] + c
            dark[(row, col)].append(img[r, c])
            dark_times[(row, col)].append(img.meta['TIME'])
    all_times = np.array(all_times)

    est_dark = {}
    for pix in dark:
        seed = ccd_bgd.aca[pix[0], pix[1]] if ccd_bgd is not None else 0
        est_dark[pix] = get_pix_estimates(dark[pix], dark_times[pix], all_times,
                                          seed=seed)

    bgs = []
    for i, img in enumerate(imgs):
        # Initialize with ccd_bgd if available
        bg = ACAImage(np.zeros((8, 8)),
                      row0=img.row0,
                      col0=img.col0)
        if ccd_bgd is not None:
            bg += ccd_bgd.aca

        for r in range(8):
            for c in range(8):
                dat_idx = (img.meta['IMGROW0'] + r,
                           img.meta['IMGCOL0'] + c)
                if dat_idx in est_dark:
                    bg[r, c] = est_dark[dat_idx][i]
        bgs.append(bg.copy())
    return bgs


def clean_imgs(dat, t_ccd=-8):
    """
    Calculate and subtract dynamic background from 'img_raw' images in an astropy
    table from the adat1 fits file.

    :param data: astropy table of adat1 file including img_raw, img_row0, and img_col0
    :param t_ccd: temperature to use for dark cal used as seed
    :returns: list of raw ACAImages with dynamic background subtracted (in DN)
    """
    imgs = []

    imgsize = dat['img_raw'].shape[1]
    if (imgsize == 4) or (imgsize == 6):
        raise ValueError("Only for 8x8 files")

    # Convert img_raw column in table to list of ACAImages
    for i, row in enumerate(dat):
        imgraw = np.ones((8, 8)) * np.nan
        imgraw[:] = np.array(row['img_raw'])
        meta = {name: row[name] for name in row.colnames}
        meta['IMGROW0'] = row['img_row0']
        meta['IMGCOL0'] = row['img_col0']
        meta['TIME'] = row['time']
        img = ACAImage(imgraw, meta=meta)
        imgs.append(img)

    # Get Dark current for seeds
    ccd_bgd = aca_dark.dark_cal.get_dark_cal_image(dat[0]['time'], t_ccd_ref=t_ccd)
    ccd_bgd = ACAImage(ccd_bgd.astype(np.float32), col0=-512, row0=-512)

    # Convert dark cal from e-/sec to DN
    ccd_bgd *= 1.696
    ccd_bgd /= 5

    # Calculate dynamic background for each img
    bgds = get_estimated_backgrounds(imgs, ccd_bgd=ccd_bgd)

    # Subtract dynamic background without oversubtraction
    bg_sub = np.ones(dat['img_raw'].shape) * np.nan
    for i, img, bgd in zip(count(), imgs, bgds):
        bg_sub[i] = np.clip(img - bgd, a_min=0, a_max=None)

    return bg_sub


def clean_file(file, t_ccd=-8):
    """
    Read raw images in DN from the 'img_raw' column of `file` and
    write back dynamic-background subtracted images in e- to 'img_corr'
    column of the same file.

    :param file: L1 ADAT file (from before calculate_centroids)
    :param t_ccd: t_ccd to be used for seed dark current
    """
    dat = Table.read(file)
    # Get cleaned images
    imgs_raw = clean_imgs(dat, t_ccd=t_ccd)
    if len(imgs_raw) == 0:
        return

    # Convert DN to e-
    imgs_corr = [img * 5 for img in imgs_raw]

    # Write out to img_corr column
    hdu = fits.open(file)
    hdu[1].data['img_corr'] = imgs_corr
    hdu.writeto(file, overwrite=True)

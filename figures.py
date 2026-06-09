# -*- coding: utf-8 -*-
"""
Drought propagation figure generation module.
Contains all figure plotting code, organized by figure number.

Each function loads the required data from pre-computed NetCDF/CSV files
and generates the corresponding figure.

Usage:
    python figures.py
    # This will generate all figures and save them to the output directory.
"""

import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors
from matplotlib import colorbar
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from scipy.stats import ttest_ind
from scipy import stats
import rasterio

# Import shared utilities
from plot_utils import (
    find_x_intersections, find_y_intersections, set_lambert_ticks,
    add_dashline, add_south_china_sea_points, get_valid_data,
    plot_spatial_trend, plot_timeseries, plot_heatmap, plot_quantile,
    style_boxplot, map_proj, gdf, xticks, yticks, nine_lines_path,
    lonMin, lonMax, latMin, latMax
)

# Import shared data loading (classify_ai used in make_figure15 and make_figure17)
from data_processing import classify_ai

plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# Output directory
OUTPUT_DIR = r'E:\drought_propagation\figure_con'
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Common extent
extent = [80, 133, 17, 54]
inset_extent = [104, 120, 1, 23]

# Common drought event colors
COLORS = ['#CDB79E', '#7AC5CD']  # dry, hot-dry


# ============================================================
# Figures 1: Trend maps and time series
# ============================================================
def make_figure1():
    """Figures 1: MK trend spatial maps and time series for P, T, SM."""
    print("Generating Figures 25-28...")

    shp = gpd.read_file(r'G:\VIwater\9大流域片\liuyu.shp')

    trend_p = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\trend_p.nc')
    trend_t = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\trend_t.nc')
    trend_sm = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\trend_sm.nc')

    pre_trend = trend_p['trend'].drop_vars('number')
    pre_p = trend_p['pvalue']
    tmp_trend = trend_t['trend'].drop_vars('number')
    tmp_p = trend_t['pvalue']
    sm_trend = trend_sm['trend'].drop_vars('number')
    sm_p = trend_sm['pvalue']

    # Load lat/lon for significance contours
    p_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\p.nc').salem.roi(shape=shp)
    lon = p_ds.longitude
    lat = p_ds.latitude

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'

    years = np.arange(1950, 2023)

    p_mean_ts = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\p_mean_ts.nc')
    t_mean_ts = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\t_mean_ts.nc')
    sm_mean_ts = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\sm_mean_ts.nc')

    # Figure 25: spatial trends
    fig = plt.figure(figsize=(12, 5))
    plt.subplots_adjust(wspace=0.2)

    ax1 = plt.subplot2grid((1, 3), (0, 0), projection=map_proj)
    ax2 = plt.subplot2grid((1, 3), (0, 1), projection=map_proj)
    ax3 = plt.subplot2grid((1, 3), (0, 2), projection=map_proj)

    plot_spatial_trend(
        fig, ax1, pre_trend, pre_p, lon, lat,
        '(a) Precipitation trend',
        cmap='bwr_r', vmin=-30, vmax=30, unit='mm a$^{-1}$',
    )

    plot_spatial_trend(
        fig, ax2, tmp_trend, tmp_p, lon, lat,
        '(b) Temperature trend',
        cmap='bwr', vmin=-0.04, vmax=0.04, unit='°C a$^{-1}$',
        ylabel='False'
    )

    plot_spatial_trend(
        fig, ax3, sm_trend, sm_p, lon, lat,
        '(c) Soil moisture trend',
        cmap='bwr_r', vmin=-0.001, vmax=0.001, unit='m$^{3}$ m$^{-3}$a$^{-1}$',
        ylabel='False', cb_ylabel='False'
    )

    fig.savefig(os.path.join(OUTPUT_DIR, 'figure1_1.png'), dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Figure 26: Precipitation time series
    fig, ax = plt.subplots(figsize=(5, 2))
    plot_timeseries(
        ax,
        '(d) Precipitation',
        years,
        p_mean_ts['tp'].values / 356,
        color='cornflowerblue',
        ylabel='mm d$^{-1}$',
        unit=' mm d$^{-1}$a$^{-1}$',
    )
    ax.set_xlabel('Year', fontsize=16)
    fig.savefig(os.path.join(OUTPUT_DIR, 'figure1_2.png'), dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Figure 27: Soil moisture time series
    fig, ax = plt.subplots(figsize=(5, 2))
    plot_timeseries(
        ax,
        '(f) Soil moisture',
        years,
        sm_mean_ts['sm'].values,
        color='limegreen',
        ylabel='m$^{3}$ m$^{-3}$',
        unit=' m$^{3}$ m$^{-3}$ a$^{-1}$',
        trend_label=False,
    )
    ax.set_xlabel('Year', fontsize=16)
    fig.savefig(os.path.join(OUTPUT_DIR, 'figure1_3.png'), dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Figure 28: Temperature time series
    fig, ax = plt.subplots(figsize=(5, 2))
    plot_timeseries(
        ax,
        '(e) Temperature',
        years,
        t_mean_ts['t2m'].values,
        color='orange',
        ylabel='°C',
        unit=' °C a$^{-1}$'
    )
    ax.set_xlabel('Year', fontsize=16)
    fig.savefig(os.path.join(OUTPUT_DIR, 'figure1_4.png'), dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("  Figures 1 saved.")


# ============================================================
# Figure 2 : 3×4 maps (6 variables × dry/hot-dry)
# ============================================================
def make_figure2():
    """Figure 2: Extended drought characteristics maps (Frequency, Ratio, Onset rate, Severity, Duration, Recovery Duration)."""
    print("Generating Figure 3...")

    hot_grid_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\hot_grid.nc')
    cold_grid_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\cold_grid.nc')

    cmaps = [plt.get_cmap("RdYlBu_r"), plt.get_cmap("RdYlBu_r"),
             plt.get_cmap('YlGnBu'), plt.get_cmap('YlOrBr'),
             plt.get_cmap('YlGnBu'), plt.get_cmap('YlGnBu')]
    norms = [colors.Normalize(vmin=0, vmax=25), colors.Normalize(vmin=0, vmax=90),
             colors.Normalize(vmin=0, vmax=2.5), colors.Normalize(vmin=0, vmax=6),
             colors.Normalize(vmin=3, vmax=5), colors.Normalize(vmin=2, vmax=3)]
    var_names = ['counts', 'ratio', 'dev_rate', 'severity', 'duration', 'rec_duration']
    titles = ['Frequency', 'Ratio', 'Onset rate', 'Severity', 'Duration', 'Recovery Duration']
    cb_title = ['Counts', '%', 'SPI month$^{-1}$', 'SPI', 'months', 'months']
    xh = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)']

    fig = plt.figure(figsize=(12, 7.5))
    gs = gridspec.GridSpec(3, 4)
    gs.update(wspace=-0.2, hspace=0.7)

    for i in range(3):
        for j in range(4):
            if j % 2 == 0:
                data_plt = abs(cold_grid_ds)
                drought_type = 'dry'
            else:
                data_plt = abs(hot_grid_ds)
                drought_type = 'hot-dry'

            ax = fig.add_subplot(gs[i, j], projection=map_proj)

            ax.add_geometries(
                gdf.geometry,
                crs=ccrs.PlateCarree(),
                edgecolor='black',
                facecolor='none',
                linewidth=0.5,
                zorder=3
            )

            add_dashline(ax, ec="black", facecolor='None', linewidth=.2, zorder=3)
            add_dashline(ax, ec="black", facecolor='None', linewidth=.4, zorder=3)

            data_plt[var_names[(i * 4 + j) // 2]].plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmaps[(i * 4 + j) // 2], norm=norms[(i * 4 + j) // 2],
                add_colorbar=False, zorder=1
            )
            ax.set_title(f'{xh[i * 4 + j]} {titles[(i * 4 + j) // 2]} ({drought_type}) ', fontsize=11)
            ax.set_extent(extent, crs=ccrs.PlateCarree())

            xlocs, xticklabels = find_x_intersections(ax, xticks)
            ax.set_xticks(xlocs)
            ax.set_xticklabels(xticklabels, fontsize=10)

            ylocs, yticklabels = find_y_intersections(ax, yticks)
            ax.set_yticks(ylocs)
            ax.set_yticklabels(yticklabels, fontsize=10)

            add_south_china_sea_points(ax)

            if j == 0:
                ax.set_ylabel('Latitude', fontsize=11)
            else:
                ax.set_ylabel('')

            if i == 2:
                ax.set_xlabel('Longitude', fontsize=11)
            else:
                ax.set_xlabel('')

            ax2 = ax.inset_axes([0.783, 0, 0.26, 0.3], projection=map_proj)
            ax2.set_extent(inset_extent, crs=ccrs.PlateCarree())

            ax2.add_geometries(gdf['geometry'], crs=ccrs.PlateCarree(),
                               edgecolor='black',
                               facecolor='None',
                               linewidth=0.4)

            data_plt[var_names[(i * 4 + j) // 2]].plot(
                ax=ax2, transform=ccrs.PlateCarree(),
                cmap=cmaps[(i * 4 + j) // 2], norm=norms[(i * 4 + j) // 2],
                add_colorbar=False, zorder=1)

            add_dashline(ax2, ec="black", facecolor='None', linewidth=0.8, zorder=1)

            if i == 2:
                if j in [0, 1]:
                    cax = fig.add_axes([0.165, 0.03, 0.33, 0.015])
                else:
                    cax = fig.add_axes([0.165 + 0.365, 0.03, 0.33, 0.015])
            else:
                if j in [0, 1]:
                    cax = fig.add_axes([0.165, 0.036 + 0.3 * (2 - i), 0.33, 0.015])
                else:
                    cax = fig.add_axes([0.165 + 0.365, 0.036 + 0.3 * (2 - i), 0.33, 0.015])

            cb = colorbar.ColorbarBase(cax,
                                       orientation='horizontal',
                                       extend='max',
                                       cmap=cmaps[(i * 4 + j) // 2],
                                       norm=norms[(i * 4 + j) // 2])
            cb.ax.tick_params(labelsize=10)
            cb.ax.set_title(cb_title[(i * 4 + j) // 2], fontsize=11, pad=0)

    fig.savefig(os.path.join(OUTPUT_DIR, 'figure2.png'), dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("  Figure 2 saved.")


# ============================================================
# Figure 3: SM anomaly maps (onset/recovery × dry/hot-dry)
# ============================================================
def make_figure3_1():
    """Figure 7: Soil moisture anomaly spatial maps during onset and recovery stages."""
    print("Generating Figure 7...")

    hot_grid_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\hot_grid.nc')
    cold_grid_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\cold_grid.nc')

    cmap = plt.get_cmap("BrBG")
    norm = colors.Normalize(vmin=-0.1, vmax=0.1)
    var_names = ['dev_sm_min', 'rec_sm_min']
    titles = ['SM anomaly in onset stage', 'SM anomaly in recovery stage']
    xh = ['(a)', '(b)', '(d)', '(e)']

    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.2, hspace=-0.1)

    for i in range(2):
        for j in range(2):
            if j == 0:
                data_plt = cold_grid_ds
                drought_type = 'dry'
            else:
                data_plt = hot_grid_ds
                drought_type = 'hot-dry'

            ax = fig.add_subplot(gs[i, j], projection=map_proj)

            ax.add_geometries(
                gdf.geometry,
                crs=ccrs.PlateCarree(),
                edgecolor='black',
                facecolor='none',
                linewidth=0.5,
                zorder=3
            )

            add_dashline(ax, ec="black", facecolor='None', linewidth=.2, zorder=3)
            add_dashline(ax, ec="black", facecolor='None', linewidth=.4, zorder=3)

            data_plt[var_names[i]].plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap, norm=norm,
                add_colorbar=False, zorder=1
            )
            ax.set_title(f'{xh[j + i * 2]} {titles[i]} ({drought_type}) ', fontsize=16)
            ax.set_extent(extent, crs=ccrs.PlateCarree())

            xlocs, xticklabels = find_x_intersections(ax, xticks)
            ax.set_xticks(xlocs)
            ax.set_xticklabels(xticklabels, fontsize=16)

            ylocs, yticklabels = find_y_intersections(ax, yticks)
            ax.set_yticks(ylocs)
            ax.set_yticklabels(yticklabels, fontsize=16)

            add_south_china_sea_points(ax)

            if j == 0:
                ax.set_ylabel('Latitude', fontsize=16)
            else:
                ax.set_ylabel('')

            if i == 1:
                ax.set_xlabel('Longitude', fontsize=16)
            else:
                ax.set_xlabel('')

            ax2 = ax.inset_axes([0.783, 0, 0.26, 0.3], projection=map_proj)
            ax2.set_extent(inset_extent, crs=ccrs.PlateCarree())

            ax2.add_geometries(gdf['geometry'], crs=ccrs.PlateCarree(),
                               edgecolor='black',
                               facecolor='None',
                               linewidth=0.4)

            data_plt[var_names[i]].plot(
                ax=ax2, transform=ccrs.PlateCarree(),
                cmap=cmap, norm=norm,
                add_colorbar=False, zorder=1)

            add_dashline(ax2, ec="black", facecolor='None', linewidth=0.8, zorder=1)

    cax = fig.add_axes([0.12, 0.07, 0.78, 0.026])
    cb = colorbar.ColorbarBase(cax,
                               orientation='horizontal',
                               extend='both',
                               cmap=cmap,
                               norm=norm)
    cb.ax.tick_params(labelsize=14)
    cb.ax.set_xlabel('m$^{3}$ m$^{-3}$', fontsize=16)

    fig.savefig(os.path.join(OUTPUT_DIR, 'figure3_1.png'), dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("  Figure 3_1 saved.")


# ============================================================
# Figure 3: SM anomaly boxplots
# ============================================================
def make_figure3_2():
    """Figure 8: SM anomaly boxplot comparison during onset and recovery."""
    print("Generating Figure 8...")

    hot_grid_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\hot_grid.nc')
    cold_grid_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\cold_grid.nc')

    data = {
        '(c) Onset': [
            get_valid_data(cold_grid_ds.dev_sm_min),
            get_valid_data(hot_grid_ds.dev_sm_min)
        ],
        '(f) Recovery': [
            get_valid_data(cold_grid_ds.rec_sm_min),
            get_valid_data(hot_grid_ds.rec_sm_min)
        ]
    }
    plt.rcParams['ytick.labelsize'] = 34
    labels = ['dry', 'hot-dry']
    plt.rcParams['font.size'] = 34

    fig, axes = plt.subplots(2, 1, figsize=(3, 16))
    fig.subplots_adjust(hspace=0.24)
    axes[0].set_ylim(-0.08, 0.05)
    for ax, (title, values) in zip(axes, data.items()):

        bp = ax.boxplot(
            values,
            widths=0.5,
            patch_artist=True,
            showfliers=False
        )

        style_boxplot(bp, COLORS)

        ax.set_title(title, fontsize=34)
        ax.set_ylabel('m$^{3}$ m$^{-3}$', fontsize=34)
        if title == '(c) Onset':
            ax.set_xticks([])
        else:
            ax.set_xticklabels(labels, rotation=45, fontsize=36)

    fig.savefig(os.path.join(OUTPUT_DIR, 'figure3_2.png'), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 3_2 saved.")

# ============================================================
# Figure 4: KDE distributions of SM anomalies
# ============================================================
def make_figure4():
    """Figure 20: KDE distributions of SM anomalies during onset and recovery stages."""
    print("Generating Figure 20...")

    hot_grid_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\hot_grid.nc')
    cold_grid_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\cold_grid.nc')

    sm_hot_onset = hot_grid_ds['dev_sm_min'].values.flatten()
    sm_hot_rec = hot_grid_ds['rec_sm_min'].values.flatten()
    sm_hot_onset = sm_hot_onset[~np.isnan(sm_hot_onset)]
    sm_hot_rec = sm_hot_rec[~np.isnan(sm_hot_rec)]

    sm_dry_onset = cold_grid_ds['dev_sm_min'].values.flatten()
    sm_dry_rec = cold_grid_ds['rec_sm_min'].values.flatten()
    sm_dry_onset = sm_dry_onset[~np.isnan(sm_dry_onset)]
    sm_dry_rec = sm_dry_rec[~np.isnan(sm_dry_rec)]

    plt.rcParams['font.size'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10

    fig, ax = plt.subplots(1, 2, figsize=(8, 2))
    fig.subplots_adjust(wspace=0.1)
    x = np.linspace(-0.12, 0.05, 300)

    # onset
    kde_hot = stats.gaussian_kde(sm_hot_onset)
    kde_dry = stats.gaussian_kde(sm_dry_onset)
    ax[0].plot(x, kde_dry(x), lw=2, label='dry', color='#ba7c29')
    ax[0].plot(x, kde_hot(x), lw=2, label='hot-dry', color='#3d9c94')
    ax[0].axvline(0, ls='--', color='gray', lw=2)
    ax[0].set_title('(a) Onset stage', fontsize=10)
    ax[0].set_xlabel('SM anomaly (m$^{3}$ m$^{-3}$)', fontsize=10)
    ax[0].set_ylabel('Probability density')
    ax[0].legend(frameon=False, fontsize=10)

    # recovery
    kde_hot = stats.gaussian_kde(sm_hot_rec)
    kde_dry = stats.gaussian_kde(sm_dry_rec)
    ax[1].plot(x, kde_hot(x), lw=2, label='hot-dry', color='#3d9c94')
    ax[1].plot(x, kde_dry(x), lw=2, label='dry', color='#ba7c29')
    ax[1].axvline(0, ls='--', color='gray')
    ax[1].set_title('(b) Recovery stage', fontsize=10)
    ax[1].set_xlabel('SM anomaly (m$^{3}$ m$^{-3}$)', fontsize=10)

    for i in range(2):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

    fig.savefig(os.path.join(OUTPUT_DIR, 'figure4.png'), dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("  Figure 4 saved.")

# ============================================================
# Figure 5: Climate variable boxplots (8-panel)
# ============================================================
def make_figure5():
    """Figure 10: 2×4 boxplots comparing climate variable anomalies between dry and hot-dry events."""
    print("Generating Figure 10...")

    hot_result = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\hot_composite.nc')
    cold_result = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\cold_composite.nc')

    vars_list = ['p', 'ssr', 'sshf', 'slhf', 'ef', 'rh', 't', 'sm']

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['ytick.labelsize'] = 12

    titles = ['(a) P', '(b) SSR', '(c) SSHF', '(d) SLHF', '(e) EF', '(f) RH', '(g) Tmax', '(h) SM']
    ylabels = ['P (mm d$^{-1}$)', 'SSR (W m$^{-2}$)', 'SSHF (W m$^{-2}$)', 'SLHF (W m$^{-2}$)',
               'EF (%)', 'RH (%)', 'Tmax (°C)', 'SM (m$^{3}$ m$^{-3}$)']
    ymax = [0.5, 48, 30, 32, 16, 5, 5, 0.04]
    ymin = [-5.5, -8, -7, -18, -32, -23, -0.8, -0.1]
    yticks_list = [[-5, -4, -3, -2, -1, 0], [0, 10, 20, 30, 40], [0, 8, 16, 24], [-10, 0, 10, 20],
                   [-30, -20, -10, 0, 10], [-18, -12, -6, 0], [0, 1, 2, 3, 4], [-0.080, -0.040, 0, 0.040]]
    ytick_labels = [['-5', '-4', '-3', '-2', '-1', '0'], ['0', '10', '20', '30', '40'],
                    ['0', '8', '16', '24'], ['-10', '0', '10', '20'],
                    ['-30', '-20', '-10', '0', '10'], ['-18', '-12', '-6', '0'],
                    ['0', '1', '2', '3', '4'], ['-0.08', '-0.04', '0', '0.04']]

    fig, axes = plt.subplots(2, 4, figsize=(9.8, 7))
    fig.subplots_adjust(wspace=0.53, hspace=0.35)
    axes = axes.flatten()

    # Compute means for reference
    a = []
    b = []
    for i, var in enumerate(vars_list):
        hot = hot_result[var].values.flatten()
        cold = cold_result[var].values.flatten()
        hot = hot[~np.isnan(hot)]
        cold = cold[~np.isnan(cold)]
        a.append(hot.mean())
        b.append(cold.mean())

    for i, var in enumerate(vars_list):
        ax = axes[i]

        hot = hot_result[var].values.flatten()
        cold = cold_result[var].values.flatten()

        hot = hot[~np.isnan(hot)]
        cold = cold[~np.isnan(cold)]

        data = [cold, hot]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.6,
            showfliers=False
        )

        style_boxplot(bp, COLORS)

        ax.set_title(titles[i], fontsize=12)
        ax.set_ylabel(ylabels[i], fontsize=12)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['dry', 'hot-dry'], fontsize=12, rotation=45)
        ax.set_ylim(ymin[i], ymax[i])
        ax.set_yticks(yticks_list[i])
        ax.set_yticklabels(ytick_labels[i])

    fig.savefig(os.path.join(OUTPUT_DIR, 'figure5.png'), dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("  Figure 5 saved.")


# ============================================================
# Figure 6: Correlation heatmaps
# ============================================================
def make_figure6():
    """Figure 6: Pearson correlation heatmaps between SM anomaly and climate variables."""
    print("Generating Figure 19...")

    hot_dev_corr = pd.read_csv(r'E:\drought_propagation\ERA5_Land_month\interp\hot_dev_corr.csv', index_col=0)
    hot_rec_corr = pd.read_csv(r'E:\drought_propagation\ERA5_Land_month\interp\hot_rec_corr.csv', index_col=0)
    cold_dev_corr = pd.read_csv(r'E:\drought_propagation\ERA5_Land_month\interp\cold_dev_corr.csv', index_col=0)
    cold_rec_corr = pd.read_csv(r'E:\drought_propagation\ERA5_Land_month\interp\cold_rec_corr.csv', index_col=0)

    hot_dev_p = pd.read_csv(r'E:\drought_propagation\ERA5_Land_month\interp\hot_dev_p.csv', index_col=0)
    hot_rec_p = pd.read_csv(r'E:\drought_propagation\ERA5_Land_month\interp\hot_rec_p.csv', index_col=0)
    cold_dev_p = pd.read_csv(r'E:\drought_propagation\ERA5_Land_month\interp\cold_dev_p.csv', index_col=0)
    cold_rec_p = pd.read_csv(r'E:\drought_propagation\ERA5_Land_month\interp\cold_rec_p.csv', index_col=0)

    hot_dev_r = hot_dev_corr.iloc[0, 1:]
    hot_dev_pv = hot_dev_p.iloc[0, 1:]
    hot_rec_r = hot_rec_corr.iloc[0, 1:]
    hot_rec_pv = hot_rec_p.iloc[0, 1:]
    cold_dev_r = cold_dev_corr.iloc[0, 1:]
    cold_dev_pv = cold_dev_p.iloc[0, 1:]
    cold_rec_r = cold_rec_corr.iloc[0, 1:]
    cold_rec_pv = cold_rec_p.iloc[0, 1:]

    hot_df = pd.DataFrame(
        [hot_dev_r.values, hot_rec_r.values],
        index=['Development', 'Recovery'],
        columns=hot_dev_r.index
    )

    cold_df = pd.DataFrame(
        [cold_dev_r.values, cold_rec_r.values],
        index=['Development', 'Recovery'],
        columns=cold_dev_r.index
    )

    hot_p_df = pd.DataFrame(
        [hot_dev_pv.values, hot_rec_pv.values],
        index=['Development', 'Recovery'],
        columns=hot_dev_pv.index
    )

    cold_p_df = pd.DataFrame(
        [cold_dev_pv.values, cold_rec_pv.values],
        index=['Development', 'Recovery'],
        columns=cold_dev_pv.index
    )

    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(2, 1, figsize=(8, 4))

    plot_heatmap(
        ax[0],
        cold_df,
        cold_p_df,
        '(a) Dry',
        xticklabel=False
    )

    im = plot_heatmap(
        ax[1],
        hot_df,
        hot_p_df,
        '(b) Hot-dry'
    )

    cax = fig.add_axes([0.99, 0.17, 0.02, 0.74])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Correlation coefficient')

    plt.tight_layout()

    fig.savefig(os.path.join(OUTPUT_DIR, 'figure6.png'), dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("  Figure 6 saved.")

# ============================================================
# Figure 7: PDP line plots
# ============================================================
def make_figure7_1():
    """Figure 7_1: Partial dependence plots for key variables (6 subplots)."""
    print("Generating Figure 12...")

    file_pairs = [
        (r'E:\drought_propagation\ERA5_Land_month\CHFD_rf\p_anom.csv', r'E:\drought_propagation\ERA5_Land_month\NHFD_rf\p_anom.csv'),
        (r'E:\drought_propagation\ERA5_Land_month\CHFD_rf\rh_anom.csv', r'E:\drought_propagation\ERA5_Land_month\NHFD_rf\rh_anom.csv'),
        (r'E:\drought_propagation\ERA5_Land_month\CHFD_rf\sshf_anom.csv', r"E:\drought_propagation\ERA5_Land_month\NHFD_rf\sshf_anom.csv"),
        (r'E:\drought_propagation\ERA5_Land_month\CHFD_rf\p.csv', r'E:\drought_propagation\ERA5_Land_month\NHFD_rf\p.csv'),
        (r'E:\drought_propagation\ERA5_Land_month\CHFD_rf\tmax.csv', r"E:\drought_propagation\ERA5_Land_month\NHFD_rf\tmax.csv"),
        (r'E:\drought_propagation\ERA5_Land_month\CHFD_rf\p_cv.csv', r"E:\drought_propagation\ERA5_Land_month\NHFD_rf\p_cv.csv")
    ]

    titles = ["(c) the Anomaly of P", "(d) the Anomaly of RH", '(e) the Anomaly of SSHF',
              '(f) P', '(g) Tmax', '(h) P_cv']

    xlabels = [
        "mm/day", "%", '(W m$^{-2}$)', "mm/day", "°C", 'mm/day'
    ]

    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18

    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.6)
    axes = axes.flatten()

    COLORS_RF = ['#1F78B4', '#6A3D9A']  # dry, hot-dry

    for i, (file1, file2) in enumerate(file_pairs):
        ax = axes[i]

        df1 = pd.read_csv(file1, index_col=0)
        df2 = pd.read_csv(file2, index_col=0)

        x1, y1 = df1["X"], df1["Y"]
        x2, y2 = df2["X"], df2["Y"]

        if i == 4:
            x1 = x1 - 273.15
            x2 = x2 - 273.15

        ax.plot(x2, y2, linewidth=3, color=COLORS_RF[0], label='dry')
        ax.plot(x1, y1, linewidth=3, color=COLORS_RF[1], label='hot-dry')

        ax.set_title(titles[i], fontsize=18.5)
        ax.set_xlabel(xlabels[i], fontsize=18.5)
        ax.set_ylabel("SM(m$^{3}$ m$^{-3}$)", fontsize=18)
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=16)

    fig.savefig(os.path.join(OUTPUT_DIR, 'figure7_1.png'), dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("  Figure 7_1 saved.")
    
# ============================================================
# Figure 7: RF importance plots
# ============================================================
def make_figure7_2():
    """Figure 7_2: Random Forest feature importance comparison (dry vs hot-dry)."""
    print("Generating Figure 16...")

    chfd = pd.read_csv(r'E:\drought_propagation\ERA5_Land_month\CHFD_rf\CHFD_importance.csv')
    nhfd = pd.read_csv(r'E:\drought_propagation\ERA5_Land_month\NHFD_rf\NHFD_importance.csv')

    plt.rcParams['xtick.labelsize'] = 14

    def plot_simple_importance(ax, df, title, color):
        y = np.arange(len(df))
        group_map = {
            'tmax_var': 'CM',
            'tmax': 'CM',
            'p_cv': 'CM',
            'p': 'CM',
            'ai': 'CM',
            'p_anom': 'HM',
            'rh_anom': 'HM',
            'slhf_anom': 'HM',
            'sshf_anom': 'HM',
            'tmax_anom': 'HM',
        }
        group_colors = {
            'CM': '#71823D',
            'HM': '#001959'
        }
        bg_colors = {
            'CM': '#E2E5D7',
            'HM': '#CBD0DD'
        }

        df['Group'] = df['feature'].map(group_map)
        group_order = ['CM', 'HM']

        for g in group_order:
            sub = df[df['Group'] == g]
            idx = sub.index
            y_pos = [np.where(df.index == i)[0][0] for i in idx]
            color_g = group_colors[g]
            ax.hlines(
                y_pos,
                0,
                sub['importance'] * 100,
                color=color_g,
                linewidth=2
            )
            ax.scatter(
                sub['importance'] * 100,
                y_pos,
                color=color_g,
                edgecolor='k',
                marker='s',
                s=50,
                zorder=3
            )
            for i in range(2):
                ax.axhspan(5 * i - 0.25, 5 * i + 4 + 0.25,
                           color=bg_colors[group_order[i]],
                           zorder=0)

        ax.set_yticks(y)
        ax.set_yticklabels(df['feature'], fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.set_xlim(-1, 70)
        ax.set_xticks([0, 20, 40, 60])

    group_map = {
        'tmax_var': 'CM',
        'tmax': 'CM',
        'p_cv': 'CM',
        'p': 'CM',
        'ai': 'CM',
        'p_anom': 'HM',
        'rh_anom': 'HM',
        'slhf_anom': 'HM',
        'sshf_anom': 'HM',
        'tmax_anom': 'HM',
    }
    group_order = ['CM', 'HM']

    chfd['Group'] = chfd['feature'].map(group_map)
    nhfd['Group'] = nhfd['feature'].map(group_map)

    chfd_sorted = pd.concat([
        chfd[chfd['Group'] == g].sort_values(by='importance')
        for g in group_order
    ])
    chfd_sorted.index = chfd_sorted['feature']
    nhfd_sorted = pd.concat([
        nhfd[nhfd['Group'] == g].sort_values(by='importance')
        for g in group_order
    ])
    nhfd_sorted.index = nhfd_sorted['feature']
    nhfd_sorted = nhfd_sorted.reindex(chfd_sorted.index)

    fig, axes = plt.subplots(1, 2, figsize=(4, 6), sharey=True)
    plot_simple_importance(axes[0], nhfd_sorted, '(a) dry', '#56B4E9')
    plot_simple_importance(axes[1], chfd_sorted, '(b) hot-dry', '#71823D')
    fig.text(0.28, -0.1, 'Importance (%IncMSE)', transform=axes[0].transAxes, fontsize=14)

    plt.tight_layout()

    fig.savefig(os.path.join(OUTPUT_DIR, 'figure7_2.png'), dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("  Figure 7_2 saved.")
    
# ============================================================
# Figure 8: AI / Land Cover quantile comparison
# ============================================================
def make_figure8():
    """Figure 8: SM anomaly quantile comparison across aridity and land cover classes
    plus classification maps (AI and Land Cover)."""
    print("Generating Figure 15...")

    # ---- Part A: AI and Land Cover maps ----
    shp = gpd.read_file(r'G:\VIwater\9大流域片\liuyu.shp')
    ai_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\ai.nc').salem.roi(shape=shp) * 0.0001
    lc_ds = xr.open_dataset(r'E:\drought_propagation\land_cover.nc').salem.roi(shape=shp)
    ai_class = classify_ai(ai_ds['ai'])
    ai_class = ai_class.drop_vars('number')
    landcover = lc_ds['landcover']

    lon = ai_ds['longitude']
    lat = ai_ds['latitude']

    ai_colors = [
        "#7f0000",  # Hyper-arid
        "#d7301f",  # Arid
        "#fdae61",  # Semi-arid
        "#d9ef8b",  # Dry sub-humid
        "#66a61e"   # Humid
    ]

    lc_colors = [
        "#006400",  # Forest
        "#32cd32",  # Shrublands
        "#CD6600",  # Savannas
        "#adff2f",  # Grasslands
        "#b0c4de",  # Barren
        "#FFC125",  # Croplands
    ]

    cmap_ai = ListedColormap(ai_colors)
    cmap_lc = ListedColormap(lc_colors)

    bounds_ai = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm_ai = BoundaryNorm(bounds_ai, ncolors=5)

    bounds_lc = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    norm_lc = BoundaryNorm(bounds_lc, ncolors=6)

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0.2, hspace=0.1)

    # AI map
    ax = fig.add_subplot(gs[0, 0], projection=map_proj)

    ax.add_geometries(
        gdf.geometry,
        crs=ccrs.PlateCarree(),
        edgecolor='black',
        facecolor='none',
        linewidth=0.5,
        zorder=3
    )

    add_dashline(ax, ec="black", facecolor='None', linewidth=.2, zorder=3)
    add_dashline(ax, ec="black", facecolor='None', linewidth=.4, zorder=3)

    ai_class.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap_ai, norm=norm_ai,
        add_colorbar=False, zorder=1
    )
    ax.set_title('(a) Aridity Index ', fontsize=14)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    xlocs, xticklabels = find_x_intersections(ax, xticks)
    ax.set_xticks(xlocs)
    ax.set_xticklabels(xticklabels, fontsize=14)

    ylocs, yticklabels = find_y_intersections(ax, yticks)
    ax.set_yticks(ylocs)
    ax.set_yticklabels(yticklabels, fontsize=14)

    add_south_china_sea_points(ax)

    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_xlabel('Longitude', fontsize=14)

    ax2 = ax.inset_axes([0.783, 0, 0.26, 0.3], projection=map_proj)
    ax2.set_extent(inset_extent, crs=ccrs.PlateCarree())

    ax2.add_geometries(gdf['geometry'], crs=ccrs.PlateCarree(),
                       edgecolor='black',
                       facecolor='None',
                       linewidth=0.4)

    ai_class.plot(
        ax=ax2, transform=ccrs.PlateCarree(),
        cmap=cmap_ai, norm=norm_ai,
        add_colorbar=False, zorder=1)

    add_dashline(ax2, ec="black", facecolor='None', linewidth=0.8, zorder=1)

    cax = fig.add_axes([0.075, -0.02, 0.4, 0.05])
    cb = colorbar.ColorbarBase(cax,
                               orientation='horizontal',
                               cmap=cmap_ai,
                               norm=norm_ai)
    cb.ax.set_xticks([1, 2, 3, 4, 5])
    cb.ax.set_xticklabels(['Hyper Arid', 'Arid', 'Semi-Arid \n (PRE/ET)', 'Dry sub-humid', 'Humid'])
    cb.ax.tick_params(labelsize=10)
    cb.ax.tick_params(axis='x', which='minor', bottom=False, top=False)

    # Land Cover map
    ax = fig.add_subplot(gs[0, 1], projection=map_proj)

    ax.add_geometries(
        gdf.geometry,
        crs=ccrs.PlateCarree(),
        edgecolor='black',
        facecolor='none',
        linewidth=0.5,
        zorder=3
    )

    add_dashline(ax, ec="black", facecolor='None', linewidth=.2, zorder=3)
    add_dashline(ax, ec="black", facecolor='None', linewidth=.4, zorder=3)

    landcover.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap_lc, norm=norm_lc,
        add_colorbar=False, zorder=1
    )
    ax.set_title('(b) Land Cover Type', fontsize=14)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    xlocs, xticklabels = find_x_intersections(ax, xticks)
    ax.set_xticks(xlocs)
    ax.set_xticklabels(xticklabels, fontsize=14)

    ylocs, yticklabels = find_y_intersections(ax, yticks)
    ax.set_yticks(ylocs)
    ax.set_yticklabels(yticklabels, fontsize=14)

    add_south_china_sea_points(ax)

    ax.set_ylabel('')
    ax.set_xlabel('Longitude', fontsize=14)

    ax2 = ax.inset_axes([0.783, 0, 0.26, 0.3], projection=map_proj)
    ax2.set_extent(inset_extent, crs=ccrs.PlateCarree())

    ax2.add_geometries(gdf['geometry'], crs=ccrs.PlateCarree(),
                       edgecolor='black',
                       facecolor='None',
                       linewidth=0.4)

    landcover.plot(
        ax=ax2, transform=ccrs.PlateCarree(),
        cmap=cmap_lc, norm=norm_lc,
        add_colorbar=False, zorder=1)

    add_dashline(ax2, ec="black", facecolor='None', linewidth=0.8, zorder=1)

    cax = fig.add_axes([0.125 + 0.38, -0.02, 0.4, 0.05])
    cb = colorbar.ColorbarBase(cax,
                               orientation='horizontal',
                               cmap=cmap_lc,
                               norm=norm_lc)
    cb.ax.set_xticks([1, 2, 3, 4, 5, 6])
    cb.ax.set_xticklabels(['Forests', 'Shrublands', 'Savannas', 'Grasslands', 'Barren', 'Croplands'])
    cb.ax.tick_params(labelsize=10)
    cb.ax.tick_params(axis='x', which='minor', bottom=False, top=False)

    fig.savefig(os.path.join(OUTPUT_DIR, 'figure15_maps.png'), dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # ---- Part B: AI and Land Cover quantile comparison ----
    df_ai_ch = pd.read_csv(r'E:\drought_propagation\ERA5_Land_month\CHFD_rf\df_ai.csv', index_col=0)
    df_ai_nh = pd.read_csv(r'E:\drought_propagation\ERA5_Land_month\NHFD_rf\df_ai.csv', index_col=0)

    df_lc_ch = pd.read_csv(r'E:\drought_propagation\ERA5_Land_month\CHFD_rf\df_lc.csv', index_col=0)
    df_lc_nh = pd.read_csv(r'E:\drought_propagation\ERA5_Land_month\NHFD_rf\df_lc.csv', index_col=0)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharey='col')
    fig.subplots_adjust(wspace=0.18, hspace=0.3)

    plot_quantile(ax[0, 0], df_ai_nh, df_ai_ch, 'ai_class', 'dev_sm_min', '(a) AI - onset')
    plot_quantile(ax[0, 1], df_ai_nh, df_ai_ch, 'ai_class', 'rec_sm_min', '(b) AI - recovery', ylabel=False)

    plot_quantile(ax[1, 0], df_lc_nh, df_lc_ch, 'landcover', 'dev_sm_min', '(c) Land Cover - onset', ai=False)
    plot_quantile(ax[1, 1], df_lc_nh, df_lc_ch, 'landcover', 'rec_sm_min', '(d) Land Cover - recovery', ai=False, ylabel=False)

    legend_elements = [
        Patch(facecolor='#CDB79E', label='dry'),
        Patch(facecolor='#7AC5CD', label='hot-dry')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=2, frameon=False, fontsize=14)

    fig.savefig(os.path.join(OUTPUT_DIR, 'figure8_quantile.png'), dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("  Figure 8 saved.")


# ============================================================
# Figure S1: Study region + AI + LC (1×3 maps)
# ============================================================
def make_figureS1():
    """Figure S1: Study region overview with DEM, Aridity Index, and Land Cover."""
    print("Generating Figure 17...")

    shp = gpd.read_file(r'G:\VIwater\9大流域片\liuyu.shp')

    ai_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\ai.nc').salem.roi(shape=shp) * 0.0001
    lc_ds = xr.open_dataset(r'E:\drought_propagation\land_cover.nc').salem.roi(shape=shp)
    ai_class = classify_ai(ai_ds['ai'])
    ai_class = ai_class.drop_vars('number')
    landcover = lc_ds['landcover']

    ai_colors = [
        "#7f0000",
        "#d7301f",
        "#fdae61",
        "#d9ef8b",
        "#66a61e"
    ]

    lc_colors = [
        "#006400",
        "#32cd32",
        "#CD6600",
        "#adff2f",
        "#b0c4de",
        "#FFC125",
    ]

    cmap_ai = ListedColormap(ai_colors)
    cmap_lc = ListedColormap(lc_colors)

    bounds_ai = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm_ai = BoundaryNorm(bounds_ai, ncolors=5)

    bounds_lc = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    norm_lc = BoundaryNorm(bounds_lc, ncolors=6)

    # Load DEM
    src = rasterio.open(r"C:\Users\msi\Downloads\dem_1km\dem_1km\dem_1km")
    data = src.read(1)
    transform = src.transform

    lon_dem = np.arange(src.width) * transform.a + transform.c
    lat_dem = np.arange(src.height) * transform.e + transform.f

    da_dem = xr.DataArray(
        data,
        coords=[("lat", lat_dem), ("lon", lon_dem)]
    )
    da_dem = da_dem.sortby('lat')
    da_dem = da_dem.where(da_dem != src.nodata)
    da_dem = da_dem.salem.roi(shape=shp)
    da_dem.astype('float32')

    dem_bounds = [-268, 0, 550, 1100, 1600, 2300, 3000, 3600, 4200, 4800, 5200, 8405]
    dem_colors = ['#AFF0E8', '#CFFBAE', '#86CD54', '#29853C', '#CBAF1D', '#BD4602',
                  '#751304', '#6C2D0C', '#94725F', '#C2C2C2', '#FEFCFF']
    cmap_dem = mcolors.ListedColormap(dem_colors)
    norm_dem = mcolors.BoundaryNorm(dem_bounds, cmap_dem.N)

    fig = plt.figure(figsize=(13, 4))
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=0.2, hspace=0.1)

    # DEM panel
    ax = fig.add_subplot(gs[0, 0], projection=map_proj)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    da_dem.plot(
        ax=ax,
        cmap=cmap_dem,
        add_colorbar=False,
        norm=norm_dem,
        transform=ccrs.PlateCarree()
    )
    ax.set_title('(a) Study region', fontsize=16)
    ax.add_geometries(shp['geometry'],
                      crs=ccrs.PlateCarree(),
                      edgecolor='black',
                      facecolor="None",
                      linewidth=0.8)
    add_dashline(ax, ec="black", facecolor='None', linewidth=.2, zorder=3)
    add_dashline(ax, ec="black", facecolor='None', linewidth=.4, zorder=3)

    xlocs, xticklabels = find_x_intersections(ax, xticks)
    ax.set_xticks(xlocs)
    ax.set_xticklabels(xticklabels, fontsize=16)

    ylocs, yticklabels = find_y_intersections(ax, yticks)
    ax.set_yticks(ylocs)
    ax.set_yticklabels(yticklabels, fontsize=16)

    add_south_china_sea_points(ax)

    ax.set_ylabel('Latitude', fontsize=16)
    ax.set_xlabel('Longitude', fontsize=16)

    ax2 = ax.inset_axes([0.783, 0, 0.26, 0.3], projection=map_proj)
    ax2.set_extent(inset_extent, crs=ccrs.PlateCarree())

    ax2.add_geometries(shp['geometry'], crs=ccrs.PlateCarree(),
                       edgecolor='black',
                       facecolor='None',
                       linewidth=0.4)

    da_dem.plot(
        ax=ax2,
        cmap=cmap_dem,
        add_colorbar=False,
        norm=norm_dem,
        transform=ccrs.PlateCarree()
    )

    add_dashline(ax2, ec="black", facecolor='None', linewidth=0.8, zorder=1)

    cax = fig.add_axes([0.09, 0, 0.27, 0.05])
    cb = colorbar.ColorbarBase(cax,
                               orientation='horizontal',
                               cmap=cmap_dem,
                               norm=norm_dem,
                               drawedges=True)
    cb.ax.set_xticks(dem_bounds)
    cb.ax.set_xticklabels(['-268', '0', '550', '1100', '1600', '2300', '3000', '3600', '4200', '4800', '5200', '8405'],
                          rotation=30)
    cb.ax.tick_params(labelsize=14)
    cb.dividers.set_color('gray')
    cb.dividers.set_linewidth(0.8)
    cb.ax.set_xlabel('(m)', fontsize=14)

    # AI panel
    ax = fig.add_subplot(gs[0, 1], projection=map_proj)

    ax.add_geometries(
        gdf.geometry,
        crs=ccrs.PlateCarree(),
        edgecolor='black',
        facecolor='none',
        linewidth=0.5,
        zorder=3
    )

    add_dashline(ax, ec="black", facecolor='None', linewidth=.2, zorder=3)
    add_dashline(ax, ec="black", facecolor='None', linewidth=.4, zorder=3)

    ai_class.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap_ai, norm=norm_ai,
        add_colorbar=False, zorder=1
    )
    ax.set_title('(b) Aridity Index ', fontsize=16)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    xlocs, xticklabels = find_x_intersections(ax, xticks)
    ax.set_xticks(xlocs)
    ax.set_xticklabels(xticklabels, fontsize=16)

    ylocs, yticklabels = find_y_intersections(ax, yticks)
    ax.set_yticks(ylocs)
    ax.set_yticklabels(yticklabels, fontsize=16)

    add_south_china_sea_points(ax)

    ax.set_ylabel('', fontsize=16)
    ax.set_xlabel('Longitude', fontsize=16)

    ax2 = ax.inset_axes([0.783, 0, 0.26, 0.3], projection=map_proj)
    ax2.set_extent(inset_extent, crs=ccrs.PlateCarree())

    ax2.add_geometries(gdf['geometry'], crs=ccrs.PlateCarree(),
                       edgecolor='black',
                       facecolor='None',
                       linewidth=0.4)

    ai_class.plot(
        ax=ax2, transform=ccrs.PlateCarree(),
        cmap=cmap_ai, norm=norm_ai,
        add_colorbar=False, zorder=1)

    add_dashline(ax2, ec="black", facecolor='None', linewidth=0.8, zorder=1)

    cax = fig.add_axes([0.37, 0, 0.26, 0.05])
    cb = colorbar.ColorbarBase(cax,
                               orientation='horizontal',
                               cmap=cmap_ai,
                               norm=norm_ai)
    cb.ax.set_xticks([1, 2, 3, 4, 5])
    cb.ax.set_xticklabels(['Hyper Arid', 'Arid', 'Semi-Arid', 'Sub-humid', 'Humid'], rotation=30)
    cb.ax.tick_params(labelsize=14)
    cb.ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    cb.ax.set_xlabel('(PRE/ET)', fontsize=14)

    # Land Cover panel
    ax = fig.add_subplot(gs[0, 2], projection=map_proj)

    ax.add_geometries(
        gdf.geometry,
        crs=ccrs.PlateCarree(),
        edgecolor='black',
        facecolor='none',
        linewidth=0.5,
        zorder=3
    )

    add_dashline(ax, ec="black", facecolor='None', linewidth=.2, zorder=3)
    add_dashline(ax, ec="black", facecolor='None', linewidth=.4, zorder=3)

    landcover.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap_lc, norm=norm_lc,
        add_colorbar=False, zorder=1
    )
    ax.set_title('(c) Land Cover Type', fontsize=16)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    xlocs, xticklabels = find_x_intersections(ax, xticks)
    ax.set_xticks(xlocs)
    ax.set_xticklabels(xticklabels, fontsize=16)

    ylocs, yticklabels = find_y_intersections(ax, yticks)
    ax.set_yticks(ylocs)
    ax.set_yticklabels(yticklabels, fontsize=16)

    add_south_china_sea_points(ax)

    ax.set_ylabel('')
    ax.set_xlabel('Longitude', fontsize=16)

    ax2 = ax.inset_axes([0.783, 0, 0.26, 0.3], projection=map_proj)
    ax2.set_extent(inset_extent, crs=ccrs.PlateCarree())

    ax2.add_geometries(gdf['geometry'], crs=ccrs.PlateCarree(),
                       edgecolor='black',
                       facecolor='None',
                       linewidth=0.4)

    landcover.plot(
        ax=ax2, transform=ccrs.PlateCarree(),
        cmap=cmap_lc, norm=norm_lc,
        add_colorbar=False, zorder=1)

    add_dashline(ax2, ec="black", facecolor='None', linewidth=0.8, zorder=1)

    cax = fig.add_axes([0.64, 0, 0.27, 0.05])
    cb = colorbar.ColorbarBase(cax,
                               orientation='horizontal',
                               cmap=cmap_lc,
                               norm=norm_lc)
    cb.ax.set_xticks([1, 2, 3, 4, 5, 6])
    cb.ax.set_xticklabels(['Forests', 'Shrublands', 'Savannas', 'Grasslands', 'Barren', 'Croplands'], rotation=30)
    cb.ax.tick_params(labelsize=14)
    cb.ax.tick_params(axis='x', which='minor', bottom=False, top=False)

    fig.savefig(os.path.join(OUTPUT_DIR, 'figureS1.png'), dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("  Figure S1 saved.")

# ============================================================
# Figure S2: Climate variable spatial maps (16-panel, 4×4)
# ============================================================
def make_figureS2():
    """Figure 11: 4×4 spatial maps of climate variable anomalies during dry/hot-dry events."""
    print("Generating Figure 11...")

    hot_result = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\hot_composite.nc')
    cold_result = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\cold_composite.nc')

    cmaps = [plt.get_cmap("RdYlBu"), plt.get_cmap('BrBG'), plt.get_cmap('BrBG_r'), plt.get_cmap('BrBG_r'),
             plt.get_cmap('BrBG'), plt.get_cmap('BrBG'), plt.get_cmap("RdYlBu_r"), plt.get_cmap('BrBG')]
    norms = [colors.Normalize(vmin=-5, vmax=5),
             colors.Normalize(vmin=-60, vmax=60),
             colors.Normalize(vmin=-30, vmax=30),
             colors.Normalize(vmin=-30, vmax=30),
             colors.Normalize(vmin=-20, vmax=20),
             colors.Normalize(vmin=-20, vmax=20),
             colors.Normalize(vmin=-8, vmax=8),
             colors.Normalize(vmin=-0.1, vmax=0.1)]
    var_names = ['p', 'ssr', 'sshf', 'slhf', 'ef', 'rh', 't', 'sm']
    titles = ['P', 'SSR', 'SSHF', 'SLHF', 'EF', 'RH', 'Tmax', 'SM']
    cb_titles = ['mm d$^{-1}$', 'W m$^{-2}$', 'W m$^{-2}$', 'W m$^{-2}$',
                 '%', '%', '°C', 'm$^{3}$ m$^{-3}$']
    xh = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)',
          '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)', '(p)']

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=-0.2, hspace=0.7)

    for i in range(4):
        for j in range(4):
            if j in [0, 2]:
                data_plt = cold_result
                drought_type = 'dry'
            else:
                data_plt = hot_result
                drought_type = 'hot-dry'

            var_name = var_names[(i * 4 + j) // 2]
            ax = fig.add_subplot(gs[i, j], projection=map_proj)

            ax.add_geometries(
                gdf.geometry,
                crs=ccrs.PlateCarree(),
                edgecolor='black',
                facecolor='none',
                linewidth=0.5,
                zorder=3
            )

            add_dashline(ax, ec="black", facecolor='None', linewidth=.2, zorder=3)
            add_dashline(ax, ec="black", facecolor='None', linewidth=.4, zorder=3)

            data_plt[var_names[(i * 4 + j) // 2]].plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmaps[(i * 4 + j) // 2], norm=norms[(i * 4 + j) // 2],
                add_colorbar=False, zorder=1
            )
            ax.set_title(f'{xh[j + i * 4]} {titles[(i * 4 + j) // 2]} ({drought_type}) ', fontsize=12)
            ax.set_extent(extent, crs=ccrs.PlateCarree())

            xlocs, xticklabels = find_x_intersections(ax, xticks)
            ax.set_xticks(xlocs)
            ax.set_xticklabels(xticklabels, fontsize=10)

            ylocs, yticklabels = find_y_intersections(ax, yticks)
            ax.set_yticks(ylocs)
            ax.set_yticklabels(yticklabels, fontsize=10)

            add_south_china_sea_points(ax)

            if j == 0:
                ax.set_ylabel('Latitude', fontsize=12)
            else:
                ax.set_ylabel('')

            if i == 3:
                ax.set_xlabel('Longitude', fontsize=12)
            else:
                ax.set_xlabel('')

            ax2 = ax.inset_axes([0.783, 0, 0.26, 0.3], projection=map_proj)
            ax2.set_extent(inset_extent, crs=ccrs.PlateCarree())

            ax2.add_geometries(gdf['geometry'], crs=ccrs.PlateCarree(),
                               edgecolor='black',
                               facecolor='None',
                               linewidth=0.4)

            data_plt[var_names[(i * 4 + j) // 2]].plot(
                ax=ax2, transform=ccrs.PlateCarree(),
                cmap=cmaps[(i * 4 + j) // 2], norm=norms[(i * 4 + j) // 2],
                add_colorbar=False, zorder=1)

            add_dashline(ax2, ec="black", facecolor='None', linewidth=0.4, zorder=1)

            if i == 3:
                if j in [0, 1]:
                    cax = fig.add_axes([0.17, 0.05, 0.33, 0.012])
                else:
                    cax = fig.add_axes([0.17 + 0.365, 0.05, 0.33, 0.012])
            else:
                if j in [0, 1]:
                    cax = fig.add_axes([0.17, 0.065 + 0.215 * (3 - i), 0.33, 0.012])
                else:
                    cax = fig.add_axes([0.17 + 0.365, 0.065 + 0.215 * (3 - i), 0.33, 0.012])

            cb = colorbar.ColorbarBase(cax,
                                       orientation='horizontal',
                                       extend='both',
                                       cmap=cmaps[(i * 4 + j) // 2],
                                       norm=norms[(i * 4 + j) // 2])
            cb.ax.tick_params(labelsize=10)
            cb.ax.set_title(cb_titles[(i * 4 + j) // 2], fontsize=10, pad=0)

    fig.savefig(os.path.join(OUTPUT_DIR, 'figureS2.png'), dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("  Figure S2 saved.")

# ============================================================
# Main: Generate all figures
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating all figures...")
    print("=" * 60)

    make_figure1()
    make_figure2()
    make_figure3_1()
    make_figure3_2()
    make_figure4()
    make_figure5()
    make_figure6()
    make_figure7_1()
    make_figure7_2()
    make_figure8()
    make_figureS1()
    make_figureS2()

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)

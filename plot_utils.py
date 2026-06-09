# -*- coding: utf-8 -*-
"""
Plotting utilities for drought propagation figures.
Contains map helper functions, common projections/settings,
and shared plotting functions used across figures.
"""

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import numpy as np
import xarray as xr
import shapely.geometry as sgeom
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import colorbar
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt
from scipy.stats import linregress
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ============================================================
# Common configuration
# ============================================================
lonMin = 80
lonMax = 133
latMin = 20
latMax = 54
xticks = list(np.arange(lonMin, lonMax + 1, 10))
yticks = list(np.arange(latMin, latMax + 1, 10))

nine_lines_path = r'G:\VIwater\中国2级流域\WGS84中国九段线国界.shp'
gdf = gpd.read_file(r'G:\VIwater\9大流域片\liuyu.shp')

map_proj = ccrs.LambertConformal(
    central_longitude=105, standard_parallels=(25, 47)
)


# ============================================================
# Map helper functions
# ============================================================
def find_x_intersections(ax, xticks):
    '''找出xticks对应的经线与下x轴的交点在data坐标下的位置和对应的ticklabel.'''
    # 获取地图的矩形边界和最大的经纬度范围.
    x0, x1, y0, y1 = ax.get_extent()
    lon0, lon1, lat0, lat1 = ax.get_extent(ccrs.PlateCarree())
    xaxis = sgeom.LineString([(x0, y0), (x1, y0)])
    # 仅选取能落入地图范围内的ticks.
    lon_ticks = [tick for tick in xticks if tick >= lon0 and tick <= lon1]

    # 每条经线有nstep个点.
    nstep = 50
    xlocs = []
    xticklabels = []
    for tick in lon_ticks:
        lon_line = sgeom.LineString(
            ax.projection.transform_points(
                ccrs.Geodetic(),
                np.full(nstep, tick),
                np.linspace(lat0, lat1, nstep)
            )[:, :2]
        )
        # 如果经线与x轴有交点,获取其位置.
        if xaxis.intersects(lon_line):
            point = xaxis.intersection(lon_line)
            xlocs.append(point.x)
            xticklabels.append(tick)
        else:
            continue

    # 用formatter添上度数和东西标识.
    formatter = LongitudeFormatter()
    xticklabels = [formatter(label) for label in xticklabels]

    return xlocs, xticklabels


def find_y_intersections(ax, yticks):
    '''找出yticks对应的纬线与左y轴的交点在data坐标下的位置和对应的ticklabel.'''
    x0, x1, y0, y1 = ax.get_extent()
    lon0, lon1, lat0, lat1 = ax.get_extent(ccrs.PlateCarree())
    yaxis = sgeom.LineString([(x0, y0), (x0, y1)])
    lat_ticks = [tick for tick in yticks if tick >= lat0 and tick <= lat1]

    nstep = 50
    ylocs = []
    yticklabels = []
    for tick in lat_ticks:
        # 注意这里与find_x_intersections的不同.
        lat_line = sgeom.LineString(
            ax.projection.transform_points(
                ccrs.Geodetic(),
                np.linspace(lon0, lon1, nstep),
                np.full(nstep, tick)
            )[:, :2]
        )
        if yaxis.intersects(lat_line):
            point = yaxis.intersection(lat_line)
            ylocs.append(point.y)
            yticklabels.append(tick)
        else:
            continue

    formatter = LatitudeFormatter()
    yticklabels = [formatter(label) for label in yticklabels]

    return ylocs, yticklabels


def set_lambert_ticks(ax, xticks, yticks):
    '''
    给一个LambertConformal投影的GeoAxes在下x轴与左y轴上添加ticks.

    要求地图边界是矩形的,即ax需要提前被set_extent方法截取成矩形.
    否则可能会出现错误.

    Parameters
    ----------
    ax : GeoAxes
        投影为LambertConformal的Axes.

    xticks : list of floats
        x轴上tick的位置.

    yticks : list of floats
        y轴上tick的位置.

    Returns
    -------
    None
    '''
    # 设置x轴.
    xlocs, xticklabels = find_x_intersections(ax, xticks)
    ax.set_xticks(xlocs)
    ax.set_xticklabels(xticklabels)
    # 设置y轴.
    ylocs, yticklabels = find_y_intersections(ax, yticks)
    ax.set_yticks(ylocs)
    ax.set_yticklabels(yticklabels)


def add_dashline(ax, **kwargs):
    '''
    在地图上画出中国省界的shapefile.

    Parameters
    ----------
    ax : GeoAxes
        目标地图.

    **kwargs
        绘制shape时用到的参数.例如linewidth,edgecolor和facecolor等.
    '''
    proj = ccrs.PlateCarree()
    reader = shpreader.Reader(nine_lines_path)
    provinces = reader.geometries()
    ax.add_geometries(provinces, proj, **kwargs)
    reader.close()


def add_south_china_sea_points(ax):
    '''在南海区域添加标记点.'''
    ax.plot(123 + 34 / 60, 25 + 45 / 60, color='k', marker='o', markersize=0.5, linewidth=0.5,
            transform=ccrs.PlateCarree(), zorder=10)
    ax.plot(123 + 40 / 60, 25 + 55 / 60, color='k', marker='o', markersize=0.5, linewidth=0.5,
            transform=ccrs.PlateCarree(), zorder=10)
    ax.plot(124 + 33 / 60, 25 + 55 / 60, color='k', marker='o', markersize=0.5, linewidth=0.5,
            transform=ccrs.PlateCarree(), zorder=10)


# ============================================================
# Data extraction helper
# ============================================================
def get_valid_data(da):
    '''提取非NaN数据.'''
    return da.values.flatten()[~np.isnan(da.values.flatten())]


# ============================================================
# Shared figure functions
# ============================================================
def plot_spatial_trend(
        fig,
        ax,
        trend,
        pvalue,
        lon,
        lat,
        title,
        cmap='bwr_r',
        vmin=-5,
        vmax=5,
        unit='mm/10a',
        extend='both',
        xlabel='True',
        ylabel='True',
        cb_ylabel='True'
):
    ax.set_extent([80, 133, 17, 54], crs=ccrs.PlateCarree())
    norm = Normalize(vmin=vmin, vmax=vmax)
    trend.plot(
        ax=ax,
        cmap=cmap,
        add_colorbar=False,
        norm=norm,
        transform=ccrs.PlateCarree()
    )

    ax.set_title(title, fontsize=12)
    ax.add_geometries(gdf['geometry'],
                      crs=ccrs.PlateCarree(),
                      edgecolor='black',
                      facecolor="None",
                      linewidth=0.8)
    add_dashline(ax, ec="black", facecolor='None', linewidth=.2, zorder=3)
    add_dashline(ax, ec="black", facecolor='None', linewidth=.4, zorder=3)
    sig = xr.where(pvalue < 0.05, 1, np.nan)
    ax.contourf(
        lon,
        lat,
        sig,
        levels=[0.5, 1.5],
        hatches=['....'],
        colors='none',
        transform=ccrs.PlateCarree()
    )
    if xlabel == 'True':
        xlocs, xticklabels = find_x_intersections(ax, xticks)
        ax.set_xticks(xlocs)
        ax.set_xticklabels(xticklabels, fontsize=12)
        ax.set_xlabel('Longitude', fontsize=12)
    else:
        ax.set_xlabel('', fontsize=12)

    add_south_china_sea_points(ax)

    if ylabel == 'True':
        ax.set_ylabel('Latitude', fontsize=12)
    else:
        ax.set_ylabel('', fontsize=12)

    ylocs, yticklabels = find_y_intersections(ax, yticks)
    ax.set_yticks(ylocs)
    ax.set_yticklabels(yticklabels, fontsize=12)

    ax2 = ax.inset_axes([0.783, 0, 0.26, 0.3], projection=map_proj)
    ax2.set_extent([104, 120, 1, 23], crs=ccrs.PlateCarree())

    ax2.add_geometries(gdf['geometry'], crs=ccrs.PlateCarree(),
                       edgecolor='black',
                       facecolor='None',
                       linewidth=0.4)

    trend.plot(
        ax=ax2,
        cmap=cmap,
        add_colorbar=False,
        norm=norm,
        transform=ccrs.PlateCarree()
    )

    ax2.contourf(
        lon,
        lat,
        sig,
        levels=[0.5, 1.5],
        hatches=['....'],
        colors='none',
        transform=ccrs.PlateCarree()
    )
    add_dashline(ax2, ec="black", facecolor='None', linewidth=0.8, zorder=1)

    pos = ax.get_position()
    # 在ax正下方创建colorbar坐标轴
    cax = fig.add_axes([
        pos.x0,          # 左边与ax对齐
        pos.y0 - 0.13,   # 下移
        pos.width,
        0.03
    ])

    cb = colorbar.ColorbarBase(cax,
                               orientation='horizontal',
                               cmap=cmap,
                               norm=norm,
                               extend=extend)

    cb.ax.tick_params(labelsize=12)
    cb.ax.set_xlabel(unit, fontsize=12, labelpad=0)
    if cb_ylabel == 'False':
        cb.ax.set_xticks([0.001, 0.000, -0.001])


def plot_timeseries(
        ax,
        title,
        years,
        data,
        color,
        ylabel,
        unit,
        trend_label=True,
        xlabel=True
):
    # 原始序列
    ax.plot(years, data, color=color, lw=3, alpha=0.7)

    # 线性趋势
    slope, intercept, r, p, std = linregress(years, data)

    trend_line = intercept + slope * years

    ax.plot(
        years,
        trend_line,
        '--',
        color='red',
        lw=3
    )
    ax.tick_params(labelsize=16)
    # 趋势文本
    if trend_label == True:
        trend_text = f'{slope * 10:.2f}{unit}'
    else:
        trend_text = f'{slope * 10:.4f}{unit}'
    ax.text(
        0.4,
        0.08,
        trend_text,
        transform=ax.transAxes,
        fontsize=16
    )
    ax.set_title(title, fontsize=16)
    if xlabel == True:
        pass
    else:
        ax.set_xticks([])

    ax.set_ylabel(ylabel, fontsize=16, labelpad=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return slope, p


def plot_heatmap(ax, data, pvalue, title, xticklabel=True):

    im = ax.imshow(
        data,
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        aspect='auto'
    )

    # ticks
    if xticklabel == True:
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_xticklabels(
            ['P', 'T', 'SSR', 'SLHF', 'SSHF', 'EF', 'RH'],
            rotation=45,
            ha='center',
            fontsize=12
        )
    else:
        ax.set_xticks([])

    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(['Onset', 'Recovery '], fontsize=12)

    # 写数字
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            r = data.iloc[i, j]
            p = pvalue.iloc[i, j]

            if p < 0.01:
                star = '**'
            elif p < 0.05:
                star = '*'
            else:
                star = ''

            text = f'{r:.2f}'
            ax.text(
                j,
                i,
                text,
                ha='center',
                va='center',
                fontsize=12
            )

            if star != '':
                ax.text(
                    j,
                    i - 0.15,      # 星号放在上方
                    star,
                    ha='center',
                    va='center',
                    fontsize=12
                )

    ax.set_title(title, fontsize=12)

    return im


def plot_quantile(ax, df_nh, df_ch, group_col, var_prefix, title, ai=True, ylabel=True):

    df_nh = df_nh[df_nh['variable'] == var_prefix]
    df_ch = df_ch[df_ch['variable'] == var_prefix]

    groups = sorted(df_nh[group_col].dropna().unique())
    x = range(len(groups))
    width = 0.15

    for i, g in enumerate(groups):

        nh = df_nh[df_nh[group_col] == g]
        ch = df_ch[df_ch[group_col] == g]

        # ---- NH ----
        ax.vlines(i - width,
                  nh[f'p5'],
                  nh[f'p95'],
                  color='#CDB79E', lw=20, alpha=0.8)

        ax.scatter(i - width,
                   nh[f'median'],
                   color='k', s=250)

        # ---- CH ----
        ax.vlines(i + width,
                  ch[f'p5'],
                  ch[f'p95'],
                  color='#7AC5CD', lw=20, alpha=0.8)

        ax.scatter(i + width,
                   ch[f'median'],
                   color='k', s=250)

    ax.set_xticks(x)
    if ai == True:
        ax.set_xticklabels(['Arid', 'Semi-Arid', 'Sub_humid', 'Humid'], fontsize=14)
    else:
        ax.set_xticklabels(['Forests', 'Savannas', 'Grasslands', 'Croplands'], fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    if ylabel == True:
        ax.set_ylabel("SM(m$^{3}$ m$^{-3}$)", fontsize=14)
    ax.tick_params(labelsize=14)


# ============================================================
# Boxplot style helper
# ============================================================
def style_boxplot(bp, colors):
    '''统一设置箱线图样式（空心 + 边框颜色）.'''
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor='none', edgecolor=colors[i % 2], linewidth=3)

    for i, median in enumerate(bp['medians']):
        median.set(color=colors[i % 2], linewidth=6)

    for i in range(len(bp['whiskers'])):
        bp['whiskers'][i].set(color=colors[i // 2], linewidth=3)
        bp['caps'][i].set(color=colors[i // 2], linewidth=3)

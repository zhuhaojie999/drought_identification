# -*- coding: utf-8 -*-
"""
Drought propagation data processing module.
Contains all data loading, computation functions, event detection,
climate variable processing, and statistical analysis.

Usage:
    python data_processing.py
    # This will run the full pipeline and save intermediate results to NetCDF/CSV.
"""

import xarray as xr
import numpy as np
import geopandas as gpd
import salem
import gc
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from collections import defaultdict
from pathlib import Path
from scipy import stats
from scipy.stats import kendalltau
from scipy.stats import linregress
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 按月份计算百分位数（经验CDF）
# ============================================================
def compute_monthly_percentile(da):
    """
    按月份计算百分位数（经验CDF）

    参数
    ----------
    da : xarray.DataArray
        (valid_time, latitude, longitude)

    返回
    ----------
    pct_da : xarray.DataArray
        百分位数 (0~1)
    """

    def _percentile_1d(x):
        """
        对一条时间序列计算百分位数（忽略NaN）
        """
        valid = np.isfinite(x)
        if valid.sum() == 0:
            return np.full_like(x, np.nan)

        x_valid = x[valid]

        # 排序
        sorted_idx = np.argsort(x_valid)
        ranks = np.empty_like(sorted_idx, dtype=float)

        # 经验CDF (rank / (n+1)) —— 避免0和1
        ranks[sorted_idx] = (np.arange(1, len(x_valid) + 1)) / (len(x_valid) + 1)

        out = np.full_like(x, np.nan, dtype=float)
        out[valid] = ranks

        return out

    # ===== 按月份分组 =====
    pct = xr.apply_ufunc(
        lambda x: _percentile_1d(x),
        da.groupby('valid_time.month'),
        input_core_dims=[['valid_time']],
        output_core_dims=[['valid_time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )

    # 保持原始时间顺序
    pct = pct.transpose('valid_time', 'latitude', 'longitude')

    return pct


# ============================================================
# 2. 单点事件识别函数
# ============================================================
def detect_events(pr_series, t_series, sm_series, i, j):
    mask = pr_series < -1
    n = len(pr_series)
    events = []

    k = 0
    while k < n:
        if mask[k]:
            start = k

            while k < n and mask[k]:
                k += 1
            end = k - 1

            duration = end - start + 1

            # 持续时间筛选
            if duration >= 2:

                pr_event = pr_series[start:end + 1]
                t_event = t_series[start:end + 1]

                # ======================
                # 基本指标
                # ======================
                severity = np.nansum(pr_event)
                peak = np.nanmin(pr_event)
                peak_idx = start + np.argmin(pr_event)

                is_hot = np.any(t_event > 0.9)

                # ======================
                # 发展速度
                # ======================
                if start > 0 and peak_idx > start:
                    dev_rate = (pr_series[peak_idx - 1] - pr_series[start - 1]) / (peak_idx - 1 - (start - 1))
                    dev_duration = peak_idx - start + 1
                else:
                    dev_rate = np.nan
                    dev_duration = 1

                # ======================
                # 恢复速度
                # ======================
                if end < n - 1 and end >= peak_idx:
                    rec_rate = (pr_series[end + 1] - pr_series[peak_idx]) / ((end + 1) - peak_idx)
                    rec_duration = end - peak_idx + 2
                else:
                    rec_rate = np.nan
                    rec_duration = 1

                dev_sm_min = np.nanmin(sm_series[start - 1:peak_idx]) if start > 0 and peak_idx > start else np.nan
                rec_sm_min = np.nanmin(sm_series[peak_idx:end + 2]) if end < n - 1 and end >= peak_idx else np.nan

                events.append({
                    "i": i,
                    "j": j,
                    "start": start,
                    "end": end,
                    "duration": duration + 2,
                    "severity": severity,
                    "peak": peak,
                    "is_hot": is_hot,
                    "dev_rate": dev_rate,
                    "rec_rate": rec_rate,
                    'dev_duration': dev_duration,
                    'rec_duration': rec_duration,
                    'dev_sm_min': dev_sm_min,
                    'rec_sm_min': rec_sm_min
                })

        else:
            k += 1

    return events


# ============================================================
# 3. 统计函数
# ============================================================
def summarize(events):
    if len(events) == 0:
        return {}

    return {
        "count": len(events),
        "mean_duration": np.mean([e["duration"] for e in events]),
        "mean_severity": np.mean([e["severity"] for e in events]),
        "mean_peak": np.mean([e["peak"] for e in events]),
        "mean_dev_rate": np.nanmean([e["dev_rate"] for e in events]),
        "mean_rec_rate": np.nanmean([e["rec_rate"] for e in events]),
    }


# ============================================================
# 4. 每个网格点平均特征
# ============================================================
def aggregate_to_grid(events, n_lat, n_lon):
    data = defaultdict(list)

    for e in events:
        key = (e["i"], e["j"])
        data[key].append(e)

    duration_map = np.full((n_lat, n_lon), np.nan)
    severity_map = np.full((n_lat, n_lon), np.nan)
    peak_map = np.full((n_lat, n_lon), np.nan)
    dev_map = np.full((n_lat, n_lon), np.nan)
    rec_map = np.full((n_lat, n_lon), np.nan)
    count_map = np.full((n_lat, n_lon), np.nan)
    devdur_map = np.full((n_lat, n_lon), np.nan)
    recdur_map = np.full((n_lat, n_lon), np.nan)
    devsm_map = np.full((n_lat, n_lon), np.nan)
    recsm_map = np.full((n_lat, n_lon), np.nan)

    for (i, j), evs in data.items():

        count_map[i, j] = len(evs)
        duration_map[i, j] = np.mean([e["duration"] for e in evs])
        severity_map[i, j] = np.mean([e["severity"] for e in evs])
        peak_map[i, j] = np.mean([e["peak"] for e in evs])

        dev_vals = [e["dev_rate"] for e in evs if not np.isnan(e["dev_rate"])]
        rec_vals = [e["rec_rate"] for e in evs if not np.isnan(e["rec_rate"])]

        dev_map[i, j] = np.mean(dev_vals) if len(dev_vals) > 0 else np.nan
        rec_map[i, j] = np.mean(rec_vals) if len(rec_vals) > 0 else np.nan

        devdur_vals = [e["dev_duration"] for e in evs if not np.isnan(e["dev_duration"])]
        recdur_vals = [e["rec_duration"] for e in evs if not np.isnan(e["rec_duration"])]

        devdur_map[i, j] = np.mean(devdur_vals) if len(devdur_vals) > 0 else np.nan
        recdur_map[i, j] = np.mean(recdur_vals) if len(recdur_vals) > 0 else np.nan

        devsm_map[i, j] = np.nanmean([e["dev_sm_min"] for e in evs])
        recsm_map[i, j] = np.nanmean([e["rec_sm_min"] for e in evs])

    return count_map, duration_map, severity_map, peak_map, dev_map, rec_map, devdur_map, recdur_map, devsm_map, recsm_map


# ============================================================
# 5. 事件编号函数
# ============================================================
def get_event_id(mask):
    mask_int = mask.astype(int)

    # 干旱开始点
    start = (mask_int == 1) & (mask_int.shift(valid_time=1, fill_value=0) == 0)

    # 累计编号
    event_id = start.cumsum("valid_time")

    # 非干旱设为 NaN
    event_id = event_id.where(mask)

    return event_id


# ============================================================
# 6. 事件合成函数
# ============================================================
def event_composite(var, event_id, is_temp=False):

    def func(x, eid):
        mask = ~np.isnan(eid)
        if mask.sum() == 0:
            return np.nan

        x = x[mask]
        eid = eid[mask]

        # 找到所有事件编号
        unique_ids = np.unique(eid)

        event_vals = []

        for uid in unique_ids:
            idx = eid == uid
            series = x[idx]

            if series.size == 0 or np.all(np.isnan(series)):
                continue

            if is_temp:
                val = np.nanmax(series)   # 温度 → 最大值
            else:
                val = np.nanmean(series)  # 其他 → 平均值

            event_vals.append(val)

        if len(event_vals) == 0:
            return np.nan

        return np.nanmean(event_vals)

    return xr.apply_ufunc(
        func,
        var,
        event_id,
        input_core_dims=[["valid_time"], ["valid_time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )


# ============================================================
# 7. RH 计算函数 (Magnus formula)
# ============================================================
def compute_rh(t_t2m, t_d2m):
    """
    计算相对湿度（Magnus公式）

    Parameters
    ----------
    t_t2m : xarray.DataArray
        气温 (K)
    t_d2m : xarray.DataArray
        露点温度 (K)

    Returns
    -------
    RH_percent : xarray.DataArray
        相对湿度 (0~100%)
    """
    Lv = 2.5e6      # J/kg
    Rv = 461        # J/(kg·K)

    RH = np.exp((Lv / Rv) * (1 / t_t2m - 1 / t_d2m))
    RH_percent = RH * 100
    RH_percent = RH_percent.clip(0, 100)

    return RH_percent


# ============================================================
# 8. EF 计算函数
# ============================================================
def compute_ef(slhf, sshf):
    """
    计算蒸发分数

    Parameters
    ----------
    slhf : xarray.DataArray
        潜热通量 (向上为正)
    sshf : xarray.DataArray
        感热通量 (向上为正)

    Returns
    -------
    EF_percent : xarray.DataArray
        蒸发分数 (0~100%)
    """
    EF = slhf / (slhf + sshf)
    EF = EF.where((slhf + sshf) != 0)
    EF_percent = EF * 100
    EF_percent = EF_percent.clip(0, 100)

    return EF_percent


# ============================================================
# 9. AI 分类函数
# ============================================================
def classify_ai(ai):
    """
    根据干旱指数进行分类

    Parameters
    ----------
    ai : xarray.DataArray
        干旱指数 (PRE/ET)

    Returns
    -------
    ai_class : xarray.DataArray
        1=Hyper-arid, 2=Arid, 3=Semi-arid, 4=Dry sub-humid, 5=Humid
    """
    ai_class = xr.full_like(ai, np.nan)

    ai_class = xr.where(ai < 0.03, 1, ai_class)
    ai_class = xr.where((ai >= 0.03) & (ai < 0.2), 2, ai_class)
    ai_class = xr.where((ai >= 0.2) & (ai < 0.5), 3, ai_class)
    ai_class = xr.where((ai >= 0.5) & (ai < 0.65), 4, ai_class)
    ai_class = xr.where(ai >= 0.65, 5, ai_class)

    return ai_class


# ============================================================
# 10. Mann-Kendall 趋势检验
# ============================================================
def mk_test_1d(y):
    """
    对一维时间序列进行 Mann-Kendall 趋势检验

    Parameters
    ----------
    y : 1D numpy array

    Returns
    -------
    slope : float
        趋势斜率（Sen's slope）
    p_value : float
        显著性p值
    """

    y = np.array(y)

    # 去除 NaN
    mask = np.isfinite(y)

    if mask.sum() < 3:
        return np.nan, np.nan

    y = y[mask]
    x = np.arange(len(y))

    # MK检验
    tau, p_value = kendalltau(x, y)

    # Sen's slope
    slopes = []

    n = len(y)

    for i in range(n - 1):
        slopes.extend((y[i + 1:] - y[i]) / (np.arange(i + 1, n) - i))

    slope = np.median(slopes)

    return slope, p_value


def calculate_mk_trend(annual_da):
    """
    对三维数据(time, lat, lon)进行MK趋势分析

    Parameters
    ----------
    annual_da : xarray.DataArray
        年尺度数据

    Returns
    -------
    xr.Dataset
        包含:
        - trend : Sen's slope
        - pvalue : 显著性
    """

    trend = xr.apply_ufunc(
        lambda y: mk_test_1d(y)[0],
        annual_da,
        input_core_dims=[['year']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )

    pvalue = xr.apply_ufunc(
        lambda y: mk_test_1d(y)[1],
        annual_da,
        input_core_dims=[['year']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )

    trend_ds = xr.Dataset({
        'trend': trend,
        'pvalue': pvalue
    })

    return trend_ds


# ============================================================
# 11. 相关系数矩阵（含p值）
# ============================================================
def corr_pvalue(df):

    cols = df.columns
    n = len(cols)

    corr = pd.DataFrame(np.zeros((n, n)),
                        columns=cols,
                        index=cols)

    pval = pd.DataFrame(np.zeros((n, n)),
                        columns=cols,
                        index=cols)

    for i in range(n):
        for j in range(n):

            r, p = pearsonr(df.iloc[:, i],
                            df.iloc[:, j])

            corr.iloc[i, j] = r
            pval.iloc[i, j] = p

    return corr, pval


# ============================================================
# 12. VPD 计算函数
# ============================================================
def compute_vpd(t2m, d2m):
    """
    计算水汽压差 (VPD)

    Parameters
    ----------
    t2m : xarray.DataArray
        2m气温 (K)
    d2m : xarray.DataArray
        2m露点温度 (K)

    Returns
    -------
    vpd : xarray.DataArray
        水汽压差 (kPa)
    """
    # 转为摄氏度
    T = t2m - 273.15
    Td = d2m - 273.15

    # 饱和水汽压 (kPa)
    es = 0.6108 * np.exp((17.27 * T) / (T + 237.3))

    # 实际水汽压 (kPa)
    ea = 0.6108 * np.exp((17.27 * Td) / (Td + 237.3))

    # VPD
    vpd = es - ea
    vpd.name = 'VPD'
    vpd.attrs['units'] = 'kPa'

    return vpd


# ============================================================
# 13. ERA5数据粗化函数
# ============================================================
def interp_era5(ds_path, factor=5):
    """
    将ERA5数据从0.1°粗化到0.5°网格

    Parameters
    ----------
    ds_path : str
        NetCDF文件路径
    factor : int
        粗化因子（默认5，即0.1°→0.5°）

    Returns
    -------
    ds_coarse : xarray.Dataset
        粗化后的数据集 (0.5°×0.5°)
    """
    ds = xr.open_dataset(ds_path)
    ds['longitude'] = xr.where(ds['longitude'] > 180, ds['longitude'] - 360, ds['longitude'])
    ds = ds.sortby('longitude')
    ds = ds.sortby('latitude')

    ds = ds.isel(latitude=slice(0, 5 * 360))  # 保留前1800个纬度点
    ds_coarse = ds.coarsen(latitude=factor, longitude=factor, boundary='trim').mean()
    del ds
    gc.collect()

    target_lat = np.arange(-89.75, 90, 0.5)  # 360个点
    target_lon = np.arange(-179.75, 180, 0.5)  # 720个点
    ds_coarse = ds_coarse.assign_coords(latitude=target_lat, longitude=target_lon)

    return ds_coarse


# ============================================================
# 14. 计算三层土壤湿度加权平均
# ============================================================
def compute_soil_moisture(sm1_path, sm2_path, sm3_path, factor=5):
    """
    从三层土壤湿度计算加权平均（0-7cm, 7-28cm, 28-100cm）

    Parameters
    ----------
    sm1_path : str
        swvl1 NetCDF路径
    sm2_path : str
        swvl2 NetCDF路径
    sm3_path : str
        swvl3 NetCDF路径
    factor : int
        粗化因子

    Returns
    -------
    sm : xarray.DataArray
        加权平均土壤湿度 (m³ m⁻³)
    """
    sm1 = interp_era5(sm1_path, factor=factor)
    sm2 = interp_era5(sm2_path, factor=factor)
    sm3 = interp_era5(sm3_path, factor=factor)

    sm = sm1['swvl1'] * 0.07 + sm2['swvl2'] * 0.21 + sm3['swvl3'] * 0.72
    sm.name = 'sm'

    return sm


# ============================================================
# 15. AI GeoTIFF重投影到ERA5网格
# ============================================================
def reproject_ai_to_era5(tif_path, lat, lon):
    """
    将Global-AI GeoTIFF重投影到ERA5的经纬度网格

    Parameters
    ----------
    tif_path : str
        AI GeoTIFF文件路径
    lat : xarray.DataArray
        ERA5纬度坐标
    lon : xarray.DataArray
        ERA5经度坐标

    Returns
    -------
    ds : xarray.Dataset
        包含 'ai' 变量的数据集，匹配ERA5网格
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())

    dst_width = len(lon)
    dst_height = len(lat)

    dst_transform = from_bounds(
        lon_min, lat_min, lon_max, lat_max,
        dst_width, dst_height
    )

    dst_data = np.empty((dst_height, dst_width), dtype=np.float32)
    reproject(
        source=data,
        destination=dst_data,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs='EPSG:4326',
        resampling=Resampling.bilinear
    )

    dst_data = np.flipud(dst_data)
    ds = xr.Dataset(
        {
            'ai': (['latitude', 'longitude'], dst_data)
        },
        coords={
            'latitude': lat,
            'longitude': lon
        }
    )

    return ds


# ============================================================
# 16. ERA5-Land数据下载（CDS API）
# ============================================================
def download_era5_monthly(variable, years=None, months=None, output_dir=None):
    """
    通过CDS API下载ERA5-Land月平均数据

    Parameters
    ----------
    variable : str
        变量名，如 "2m_temperature", "total_precipitation",
        "volumetric_soil_water_layer_1" 等
    years : list of str, optional
        年份列表，默认 1950-2022
    months : list of str, optional
        月份列表，默认 01-12
    output_dir : str, optional
        输出目录，默认当前目录

    Notes
    -----
    需要预先安装并配置CDS API:
        pip install cdsapi
        并在 ~/.cdsapirc 中配置URL和Key
    """
    import cdsapi

    if years is None:
        years = [str(y) for y in range(1950, 2023)]
    if months is None:
        months = [f"{m:02d}" for m in range(1, 13)]

    dataset = "reanalysis-era5-land-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [variable],
        "year": years,
        "month": months,
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    client = cdsapi.Client()
    result = client.retrieve(dataset, request)

    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        result.download(os.path.join(output_dir, f"{variable}.nc"))
    else:
        result.download()

    print(f"Downloaded: {variable}")


# ============================================================
# ============================================================
# MAIN PROCESSING PIPELINE
# ============================================================
# ============================================================
if __name__ == "__main__":

    # ========================================================
    # Part A: Drought Event Detection
    # ========================================================
    print("=" * 60)
    print("Part A: Drought Event Detection")
    print("=" * 60)

    shp = gpd.read_file(r'G:\VIwater\9大流域片\liuyu.shp')

    sm = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\sm.nc').salem.roi(shape=shp)
    p = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\p.nc').salem.roi(shape=shp)
    t = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\t.nc').salem.roi(shape=shp)

    t_pct = compute_monthly_percentile(t)

    monthly_mean = p.groupby('valid_time.month').mean(dim='valid_time', skipna=True)
    monthly_std = p.groupby('valid_time.month').std(dim='valid_time', skipna=True)

    monthly_std = monthly_std.where(monthly_std != 0)

    z_score = p.groupby('valid_time.month') - monthly_mean
    z_score = z_score.groupby('valid_time.month') / monthly_std

    sm_mean = sm.groupby('valid_time.month').mean(dim='valid_time', skipna=True)
    sm_ano = sm.groupby('valid_time.month') - sm_mean

    ds = xr.Dataset({
        'pr_z': z_score.tp,
        't_pct': t_pct.t2m,
        'sm_ano': sm_ano.sm})

    pr = ds.pr_z
    t_pct = ds.t_pct
    sm = ds.sm_ano

    time = ds.valid_time
    lat = pr.latitude
    lon = pr.longitude

    n_time = pr.sizes['valid_time']
    n_lat = len(lat)
    n_lon = len(lon)

    # ======================
    # 输出容器
    # ======================
    hot_mask = xr.full_like(pr, np.nan)
    cold_mask = xr.full_like(pr, np.nan)

    hot_events = []
    cold_events = []

    # ======================
    # 遍历网格
    # ======================
    print("  Traversing grid for event detection...")
    for i in range(len(lat)):
        for j in range(len(lon)):

            pr_series = pr[:, i, j].values
            t_series = t_pct[:, i, j].values
            sm_series = sm[:, i, j].values

            if np.all(np.isnan(pr_series)):
                continue

            events = detect_events(pr_series, t_series, sm_series, i, j)

            for ev in events:
                s, e = ev["start"], ev["end"]

                if ev["is_hot"]:
                    hot_events.append(ev)
                    hot_mask[s:e + 1, i, j] = pr_series[s:e + 1]
                else:
                    cold_events.append(ev)
                    cold_mask[s:e + 1, i, j] = pr_series[s:e + 1]

    hot_summary = summarize(hot_events)
    cold_summary = summarize(cold_events)

    print("高温干旱统计：", hot_summary)
    print("非高温干旱统计：", cold_summary)

    hot_maps = aggregate_to_grid(hot_events, n_lat, n_lon)
    cold_maps = aggregate_to_grid(cold_events, n_lat, n_lon)

    hot_grid_ds = xr.Dataset({
        "counts": (("latitude", "longitude"), hot_maps[0]),
        "duration": (("latitude", "longitude"), hot_maps[1]),
        "severity": (("latitude", "longitude"), hot_maps[2]),
        "peak": (("latitude", "longitude"), hot_maps[3]),
        "dev_rate": (("latitude", "longitude"), hot_maps[4]),
        "rec_rate": (("latitude", "longitude"), hot_maps[5]),
        "dev_duration": (("latitude", "longitude"), hot_maps[6]),
        "rec_duration": (("latitude", "longitude"), hot_maps[7]),
        "dev_sm_min": (("latitude", "longitude"), hot_maps[8]),
        "rec_sm_min": (("latitude", "longitude"), hot_maps[9]),
    }, coords={"latitude": lat, "longitude": lon})

    cold_grid_ds = xr.Dataset({
        "counts": (("latitude", "longitude"), cold_maps[0]),
        "duration": (("latitude", "longitude"), cold_maps[1]),
        "severity": (("latitude", "longitude"), cold_maps[2]),
        "peak": (("latitude", "longitude"), cold_maps[3]),
        "dev_rate": (("latitude", "longitude"), cold_maps[4]),
        "rec_rate": (("latitude", "longitude"), cold_maps[5]),
        "dev_duration": (("latitude", "longitude"), cold_maps[6]),
        "rec_duration": (("latitude", "longitude"), cold_maps[7]),
        "dev_sm_min": (("latitude", "longitude"), cold_maps[8]),
        "rec_sm_min": (("latitude", "longitude"), cold_maps[9]),
    }, coords={"latitude": lat, "longitude": lon})

    # 计算比例
    hot_count = hot_grid_ds["counts"]
    cold_count = cold_grid_ds["counts"]
    total_count = hot_count + cold_count

    hot_ratio = xr.where(total_count > 0, hot_count / total_count, np.nan)
    cold_ratio = xr.where(total_count > 0, cold_count / total_count, np.nan)

    hot_grid_ds["ratio"] = hot_ratio * 100
    cold_grid_ds["ratio"] = cold_ratio * 100

    # 保存事件mask
    hot_ds = xr.Dataset({"hot_drought": hot_mask})
    cold_ds = xr.Dataset({"cold_drought": cold_mask})

    hot_ds.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\hot_map.nc')
    cold_ds.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\cold_map.nc')

    # 保存网格统计结果
    hot_grid_ds.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\hot_grid.nc')
    cold_grid_ds.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\cold_grid.nc')

    print("  Saved hot_map.nc, cold_map.nc, hot_grid.nc, cold_grid.nc")

    # ========================================================
    # Part B: Climate Variable Composites
    # ========================================================
    print("\n" + "=" * 60)
    print("Part B: Climate Variable Composites")
    print("=" * 60)

    # Load saved masks
    hot_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\hot_map.nc')
    cold_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\cold_map.nc')

    p_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\p.nc').salem.roi(shape=shp) * 1000
    p_ano = p_ds.groupby('valid_time.month') - p_ds.groupby('valid_time.month').mean(dim='valid_time', skipna=True)

    dewpoint_t = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\dewpoint_t.nc').salem.roi(shape=shp)
    t_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\t.nc').salem.roi(shape=shp)
    t_ano = t_ds.groupby('valid_time.month') - t_ds.groupby('valid_time.month').mean(dim='valid_time', skipna=True)

    ssr = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\ssr.nc').salem.roi(shape=shp) / 86400
    ssr_ano = ssr.groupby('valid_time.month') - ssr.groupby('valid_time.month').mean(dim='valid_time', skipna=True)

    slhf = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\slhf.nc').salem.roi(shape=shp) / 86400
    slhf_ano = slhf.groupby('valid_time.month') - slhf.groupby('valid_time.month').mean(dim='valid_time', skipna=True)

    sshf = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\sshf.nc').salem.roi(shape=shp) / 86400
    sshf_ano = sshf.groupby('valid_time.month') - sshf.groupby('valid_time.month').mean(dim='valid_time', skipna=True)

    sm_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\sm.nc').salem.roi(shape=shp)
    sm_ano = sm_ds.groupby('valid_time.month') - sm_ds.groupby('valid_time.month').mean(dim='valid_time', skipna=True)

    # 计算 RH
    T = t_ds['t2m']
    Td = dewpoint_t['d2m']
    RH_percent = compute_rh(T, Td)
    rh = xr.Dataset({"rh": RH_percent})
    rh_ano = rh.groupby('valid_time.month') - rh.groupby('valid_time.month').mean(dim='valid_time', skipna=True)

    # 计算 EF
    slhf_up = slhf.slhf
    sshf_up = sshf.sshf
    EF_percent = compute_ef(slhf_up, sshf_up)
    ef = xr.Dataset({"ef": EF_percent})
    ef_ano = ef.groupby('valid_time.month') - ef.groupby('valid_time.month').mean(dim='valid_time', skipna=True)

    vars_dict = {
        "p": p_ano.tp,
        "t": t_ano.t2m,
        "ssr": ssr_ano.ssr,
        "slhf": slhf_ano.slhf,
        "sshf": -sshf_ano.sshf,
        "ef": ef_ano.ef,
        "rh": rh_ano.rh,
        "sm": sm_ano.sm
    }

    hot_bool = hot_ds["hot_drought"].notnull()
    cold_bool = cold_ds["cold_drought"].notnull()

    hot_event_id = get_event_id(hot_bool)
    cold_event_id = get_event_id(cold_bool)

    hot_result = xr.Dataset()
    cold_result = xr.Dataset()

    for name, da in vars_dict.items():
        hot_result[name] = event_composite(
            da, hot_event_id,
            is_temp=(name == "t")
        )
        cold_result[name] = event_composite(
            da, cold_event_id,
            is_temp=(name == "t")
        )

    # 保存合成结果
    hot_result.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\hot_composite.nc')
    cold_result.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\cold_composite.nc')

    print("  Saved hot_composite.nc, cold_composite.nc")

    # ========================================================
    # Part C: Land Cover Processing
    # ========================================================
    print("\n" + "=" * 60)
    print("Part C: Land Cover Processing")
    print("=" * 60)

    with rasterio.open(r'C:\Users\msi\Downloads\MCD12C1_2015_China.tif') as src:
        data = src.read(1)
        transform = src.transform

    nlat_lc, nlon_lc = data.shape

    lon_lc = np.arange(nlon_lc) * transform.a + transform.c
    lat_lc = np.arange(nlat_lc) * transform.e + transform.f

    if lat_lc[0] > lat_lc[-1]:
        lat_lc = lat_lc[::-1]
        data = data[::-1, :]

    data = data.astype(float)

    new = np.full_like(data, np.nan)

    # 分类规则
    new[(data >= 1) & (data <= 5)] = 1
    new[(data >= 6) & (data <= 7)] = 2
    new[(data >= 8) & (data <= 9)] = 3
    new[data == 10] = 4
    new[(data == 15) | (data == 16)] = 5
    new[(data == 12) | (data == 14)] = 6
    new[(data == 11) | (data == 13)] = 7

    # 0 → NaN
    new[data == 0] = np.nan

    da = xr.DataArray(
        new,
        coords={'lat': lat_lc, 'lon': lon_lc},
        dims=['lat', 'lon'],
        name='landcover'
    )

    da_out = da.interp(
        lat=hot_result.latitude,
        lon=hot_result.longitude,
        method='nearest'
    )

    da_out.to_netcdf(r'E:\drought_propagation\land_cover.nc')
    print("  Saved land_cover.nc")

    # ========================================================
    # Part D: Background Climate
    # ========================================================
    print("\n" + "=" * 60)
    print("Part D: Background Climate")
    print("=" * 60)

    p_bg = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\p.nc').salem.roi(shape=shp) * 1000
    t_bg = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\t.nc').salem.roi(shape=shp)
    sm_bg = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\sm.nc').salem.roi(shape=shp)
    ai = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\ai.nc').salem.roi(shape=shp) * 0.0001

    pr_year_mean = p_bg.tp.resample(valid_time='1Y').mean()
    pr_cli = pr_year_mean.mean(dim='valid_time')
    std = pr_year_mean.std(dim='valid_time')
    pr_cv = std / pr_cli

    sm_year_mean = sm_bg.sm.resample(valid_time='1Y').mean()
    sm_cli = sm_year_mean.mean(dim='valid_time')

    tmax = t_bg['t2m'].resample(valid_time='1Y').max()
    tmax_mean = tmax.mean(dim='valid_time')
    tmax_var = tmax.var(dim='valid_time')

    pr_cli.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\pr_cli.nc')
    pr_cv.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\pr_cv.nc')
    tmax_mean.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\tmax_mean.nc')
    tmax_var.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\tmax_var.nc')

    print("  Saved pr_cli.nc, pr_cv.nc, tmax_mean.nc, tmax_var.nc")

    # ========================================================
    # Part E: AI Classification
    # ========================================================
    print("\n" + "=" * 60)
    print("Part E: AI Classification")
    print("=" * 60)

    ai_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\ai.nc').salem.roi(shape=shp) * 0.0001
    ai_class = classify_ai(ai_ds['ai'])
    ai_class = ai_class.drop_vars('number')
    ai_class.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\ai_class.nc')

    print("  Saved ai_class.nc")

    # ========================================================
    # Part F: MK Trend Analysis
    # ========================================================
    print("\n" + "=" * 60)
    print("Part F: MK Trend Analysis")
    print("=" * 60)

    p_mk = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\p.nc').salem.roi(shape=shp) * 1000
    t_mk = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\t.nc').salem.roi(shape=shp) - 273.15
    sm_mk = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\sm.nc').salem.roi(shape=shp)

    annual_p = p_mk['tp'].groupby('valid_time.year').mean(dim='valid_time') * 12 * 365
    annual_t = t_mk['t2m'].groupby('valid_time.year').mean(dim='valid_time')
    annual_sm = sm_mk['sm'].groupby('valid_time.year').mean(dim='valid_time')

    p_mean = annual_p.mean(dim=['latitude', 'longitude'], skipna=True)
    t_mean = annual_t.mean(dim=['latitude', 'longitude'], skipna=True)
    sm_mean = annual_sm.mean(dim=['latitude', 'longitude'], skipna=True)

    trend_p = calculate_mk_trend(annual_p)
    trend_t = calculate_mk_trend(annual_t)
    trend_sm = calculate_mk_trend(annual_sm)

    trend_p.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\trend_p.nc')
    trend_t.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\trend_t.nc')
    trend_sm.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\trend_sm.nc')

    # 保存时间序列均值
    p_mean.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\p_mean_ts.nc')
    t_mean.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\t_mean_ts.nc')
    sm_mean.to_netcdf(r'E:\drought_propagation\ERA5_Land_month\interp\sm_mean_ts.nc')

    print("  Saved trend_p.nc, trend_t.nc, trend_sm.nc")
    print("  Saved p_mean_ts.nc, t_mean_ts.nc, sm_mean_ts.nc")

    # ========================================================
    # Part G: Land Cover / AI Classification Statistics
    # ========================================================
    print("\n" + "=" * 60)
    print("Part G: Land Cover / AI Classification Statistics")
    print("=" * 60)

    land_cover = xr.open_dataset(r'E:\drought_propagation\land_cover.nc').salem.roi(shape=shp)
    landcover = land_cover['landcover']

    # 重新加载 grids（确保使用已保存的最新版本）
    hot_grid_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\hot_grid.nc')
    cold_grid_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\cold_grid.nc')
    hot_result = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\hot_composite.nc')
    cold_result = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\cold_composite.nc')
    ai_class = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\ai_class.nc')

    lc_list = [1, 3, 4, 6]
    ai_list = [2, 3, 4, 5]

    # ---- Land cover × variables (hot) ----
    results = []
    for lc in lc_list:
        mask = (landcover == lc)
        for var_name in ['p', 'ssr', 'sshf', 'slhf', 'ef', 'rh', 't', 'sm']:
            data = hot_result[var_name].where(mask)
            mean_val = data.mean(dim=["latitude", "longitude"], skipna=True)
            results.append({
                "landcover": lc,
                "variable": var_name,
                "mean": float(mean_val.values),
            })
    df_lc = pd.DataFrame(results)
    df_lc.to_csv(r'E:\drought_propagation\ERA5_Land_month\CHFD_rf\df_lc_pro.csv', index=False)

    # ---- Land cover × variables (cold) ----
    results = []
    for lc in lc_list:
        mask = (landcover == lc)
        for var_name in ['p', 'ssr', 'sshf', 'slhf', 'ef', 'rh', 't', 'sm']:
            data = cold_result[var_name].where(mask)
            mean_val = data.mean(dim=["latitude", "longitude"], skipna=True)
            results.append({
                "landcover": lc,
                "variable": var_name,
                "mean": float(mean_val.values),
            })
    df_lc = pd.DataFrame(results)
    df_lc.to_csv(r'E:\drought_propagation\ERA5_Land_month\NHFD_rf\df_lc_pro.csv', index=False)

    # ---- Land cover × SM (cold) ----
    results = []
    for lc in lc_list:
        mask = (landcover == lc)
        for var_name in ["dev_sm_min", "rec_sm_min"]:
            data = cold_grid_ds[var_name].where(mask)
            mean_val = data.mean(dim=["latitude", "longitude"], skipna=True)
            q = data.quantile(
                q=[0.05, 0.25, 0.5, 0.75, 0.95],
                dim=["latitude", "longitude"],
                skipna=True
            )
            results.append({
                "landcover": lc,
                "variable": var_name,
                "mean": float(mean_val.values),
                "p5": float(q.sel(quantile=0.05).values),
                "q1": float(q.sel(quantile=0.25).values),
                "median": float(q.sel(quantile=0.5).values),
                "q3": float(q.sel(quantile=0.75).values),
                "p95": float(q.sel(quantile=0.95).values)
            })
    df_lc = pd.DataFrame(results)
    df_lc.to_csv(r'E:\drought_propagation\ERA5_Land_month\NHFD_rf\df_lc.csv', index=False)

    # ---- Land cover × SM (hot) ----
    results = []
    for lc in lc_list:
        mask = (landcover == lc)
        for var_name in ["dev_sm_min", "rec_sm_min"]:
            data = hot_grid_ds[var_name].where(mask)
            mean_val = data.mean(dim=["latitude", "longitude"], skipna=True)
            q = data.quantile(
                q=[0.05, 0.25, 0.5, 0.75, 0.95],
                dim=["latitude", "longitude"],
                skipna=True
            )
            results.append({
                "landcover": lc,
                "variable": var_name,
                "mean": float(mean_val.values),
                "p5": float(q.sel(quantile=0.05).values),
                "q1": float(q.sel(quantile=0.25).values),
                "median": float(q.sel(quantile=0.5).values),
                "q3": float(q.sel(quantile=0.75).values),
                "p95": float(q.sel(quantile=0.95).values)
            })
    df_lc = pd.DataFrame(results)
    df_lc.to_csv(r'E:\drought_propagation\ERA5_Land_month\CHFD_rf\df_lc.csv', index=False)

    # ---- AI × SM (cold) ----
    results = []
    for ai_val in ai_list:
        mask = (ai_class == ai_val)
        for var_name in ["dev_sm_min", "rec_sm_min"]:
            data = cold_grid_ds[var_name].where(mask)
            mean_val = data.mean(dim=["latitude", "longitude"], skipna=True)
            q = data.quantile(
                q=[0.05, 0.25, 0.5, 0.75, 0.95],
                dim=["latitude", "longitude"],
                skipna=True
            )
            results.append({
                "ai_class": ai_val,
                "variable": var_name,
                "mean": float(mean_val.values),
                "p5": float(q.sel(quantile=0.05).values),
                "q1": float(q.sel(quantile=0.25).values),
                "median": float(q.sel(quantile=0.5).values),
                "q3": float(q.sel(quantile=0.75).values),
                "p95": float(q.sel(quantile=0.95).values)
            })
    df_ai = pd.DataFrame(results)
    df_ai.to_csv(r'E:\drought_propagation\ERA5_Land_month\NHFD_rf\df_ai.csv', index=False)

    # ---- AI × SM (hot) ----
    results = []
    for ai_val in ai_list:
        mask = (ai_class == ai_val)
        for var_name in ["dev_sm_min", "rec_sm_min"]:
            data = hot_grid_ds[var_name].where(mask)
            mean_val = data.mean(dim=["latitude", "longitude"], skipna=True)
            q = data.quantile(
                q=[0.05, 0.25, 0.5, 0.75, 0.95],
                dim=["latitude", "longitude"],
                skipna=True
            )
            results.append({
                "ai_class": ai_val,
                "variable": var_name,
                "mean": float(mean_val.values),
                "p5": float(q.sel(quantile=0.05).values),
                "q1": float(q.sel(quantile=0.25).values),
                "median": float(q.sel(quantile=0.5).values),
                "q3": float(q.sel(quantile=0.75).values),
                "p95": float(q.sel(quantile=0.95).values)
            })
    df_ai = pd.DataFrame(results)
    df_ai.to_csv(r'E:\drought_propagation\ERA5_Land_month\CHFD_rf\df_ai.csv', index=False)

    print("  Saved df_lc.csv, df_ai.csv for both CHFD and NHFD")

    # ========================================================
    # Part H: Correlation Computation
    # ========================================================
    print("\n" + "=" * 60)
    print("Part H: Correlation Computation")
    print("=" * 60)

    hot_dev_vars = {
        'dev_sm_min': hot_grid_ds['dev_sm_min'].values.flatten(),
        'p': hot_result['p'].values.flatten(),
        't': hot_result['t'].values.flatten(),
        'ssr': hot_result['ssr'].values.flatten(),
        'slhf': hot_result['slhf'].values.flatten(),
        'sshf': hot_result['sshf'].values.flatten(),
        'ef': hot_result['ef'].values.flatten(),
        'rh': hot_result['rh'].values.flatten(),
    }

    hot_rec_vars = {
        'rec_sm_min': hot_grid_ds['rec_sm_min'].values.flatten(),
        'p': hot_result['p'].values.flatten(),
        't': hot_result['t'].values.flatten(),
        'ssr': hot_result['ssr'].values.flatten(),
        'slhf': hot_result['slhf'].values.flatten(),
        'sshf': hot_result['sshf'].values.flatten(),
        'ef': hot_result['ef'].values.flatten(),
        'rh': hot_result['rh'].values.flatten(),
    }

    cold_dev_vars = {
        'dev_sm_min': cold_grid_ds['dev_sm_min'].values.flatten(),
        'p': cold_result['p'].values.flatten(),
        't': cold_result['t'].values.flatten(),
        'ssr': cold_result['ssr'].values.flatten(),
        'slhf': cold_result['slhf'].values.flatten(),
        'sshf': cold_result['sshf'].values.flatten(),
        'ef': cold_result['ef'].values.flatten(),
        'rh': cold_result['rh'].values.flatten(),
    }

    cold_rec_vars = {
        'rec_sm_min': cold_grid_ds['rec_sm_min'].values.flatten(),
        'p': cold_result['p'].values.flatten(),
        't': cold_result['t'].values.flatten(),
        'ssr': cold_result['ssr'].values.flatten(),
        'slhf': cold_result['slhf'].values.flatten(),
        'sshf': cold_result['sshf'].values.flatten(),
        'ef': cold_result['ef'].values.flatten(),
        'rh': cold_result['rh'].values.flatten(),
    }

    hot_dev_df = pd.DataFrame(hot_dev_vars).dropna()
    hot_rec_df = pd.DataFrame(hot_rec_vars).dropna()
    cold_dev_df = pd.DataFrame(cold_dev_vars).dropna()
    cold_rec_df = pd.DataFrame(cold_rec_vars).dropna()

    hot_dev_corr, hot_dev_p = corr_pvalue(hot_dev_df)
    hot_rec_corr, hot_rec_p = corr_pvalue(hot_rec_df)
    cold_dev_corr, cold_dev_p = corr_pvalue(cold_dev_df)
    cold_rec_corr, cold_rec_p = corr_pvalue(cold_rec_df)

    hot_dev_corr.to_csv(r'E:\drought_propagation\ERA5_Land_month\interp\hot_dev_corr.csv')
    hot_rec_corr.to_csv(r'E:\drought_propagation\ERA5_Land_month\interp\hot_rec_corr.csv')
    cold_dev_corr.to_csv(r'E:\drought_propagation\ERA5_Land_month\interp\cold_dev_corr.csv')
    cold_rec_corr.to_csv(r'E:\drought_propagation\ERA5_Land_month\interp\cold_rec_corr.csv')

    hot_dev_p.to_csv(r'E:\drought_propagation\ERA5_Land_month\interp\hot_dev_p.csv')
    hot_rec_p.to_csv(r'E:\drought_propagation\ERA5_Land_month\interp\hot_rec_p.csv')
    cold_dev_p.to_csv(r'E:\drought_propagation\ERA5_Land_month\interp\cold_dev_p.csv')
    cold_rec_p.to_csv(r'E:\drought_propagation\ERA5_Land_month\interp\cold_rec_p.csv')

    print("  Saved correlation matrices")

    print("\n" + "=" * 60)
    print("All data processing complete!")
    print("=" * 60)

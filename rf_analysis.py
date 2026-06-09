# -*- coding: utf-8 -*-
"""
Random Forest analysis module for drought propagation study.
Contains RF training, partial dependence computation,
feature importance visualization, and importance bar plots.

Usage:
    python rf_analysis.py
    # This will train RF models, compute PDP, and generate importance plots.
"""

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

# Import shared utilities
from plot_utils import gdf

plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = r'E:\drought_propagation\figure_con'
RF_OUTPUT_DIR = r'E:\drought_propagation\ERA5_Land_month'
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(RF_OUTPUT_DIR, 'CHFD_rf'), exist_ok=True)
os.makedirs(os.path.join(RF_OUTPUT_DIR, 'NHFD_rf'), exist_ok=True)


# ============================================================
# 1. RF Training Function
# ============================================================
def train_rf(X, y, n_estimators=300):
    """
    Train a Random Forest Regressor and evaluate on a held-out test set.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    n_estimators : int
        Number of trees.

    Returns
    -------
    model : RandomForestRegressor
        Trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f'R² = {r2:.3f}')

    return model


# ============================================================
# 2. Importance Bar Plot Function
# ============================================================
def plot_simple_importance(ax, df, title, color):
    """
    Draw a horizontal importance bar plot with CM/HM grouping.

    Parameters
    ----------
    ax : matplotlib.Axes
        Target axes.
    df : pd.DataFrame
        Must have 'feature' and 'importance' columns.
    title : str
        Subplot title.
    color : str
        Not used directly (colors come from group_colors).
    """
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


# ============================================================
# ============================================================
# MAIN RF ANALYSIS PIPELINE
# ============================================================
# ============================================================
if __name__ == "__main__":

    # ========================================================
    # Step 1: Prepare RF input data
    # ========================================================
    print("=" * 60)
    print("Step 1: Preparing RF input data...")
    print("=" * 60)

    shp = gpd.read_file(r'G:\VIwater\9大流域片\liuyu.shp')

    # Load processed data
    hot_grid_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\hot_grid.nc')
    hot_result = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\hot_composite.nc')
    cold_grid_ds = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\cold_grid.nc')
    cold_result = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\cold_composite.nc')

    # Load background climate
    pr_cli = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\pr_cli.nc')
    pr_cv = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\pr_cv.nc')
    tmax_mean = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\tmax_mean.nc')
    tmax_var = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\tmax_var.nc')

    ai = xr.open_dataset(r'E:\drought_propagation\ERA5_Land_month\interp\ai.nc').salem.roi(shape=shp) * 0.0001
    land_cover = xr.open_dataset(r'E:\drought_propagation\land_cover.nc').salem.roi(shape=shp)

    # ========================================================
    # Step 2: Train RF for Hot-dry (CHMD) events
    # ========================================================
    print("\n" + "=" * 60)
    print("Step 2: Training RF for hot-dry (CHMD) events...")
    print("=" * 60)

    mask = land_cover.landcover
    y = hot_grid_ds.rec_sm_min.where(mask)

    features = [
        'p_anom', 'sshf_anom', 'slhf_anom', 'ssr_ano',
        'rh_anom', 'tmax_anom',
        'p', 'p_cv', 'tmax', 'tmax_var', 'ai'
    ]

    X = xr.Dataset({
        'p_anom': hot_result.p.where(mask),
        'sshf_anom': hot_result.sshf.where(mask),
        'slhf_anom': hot_result.slhf.where(mask),
        'ssr_ano': hot_result.ssr.where(mask),
        'rh_anom': hot_result.rh.where(mask),
        'tmax_anom': hot_result.t.where(mask),
        'p': pr_cli.where(mask),
        'p_cv': pr_cv.where(mask),
        'tmax': tmax_mean.where(mask),
        'tmax_var': tmax_var.where(mask),
        'ai': ai.ai.where(mask)})

    # Flatten to DataFrame
    df = xr.merge([X, y]).to_dataframe().dropna()

    X_df = df[features]
    y_df = df['rec_sm_min']

    rf_chmd = train_rf(X_df, y_df, n_estimators=300)

    # Test different n_estimators
    for n in [200, 300, 400, 500]:
        print(f"n_estimators = {n}")
        train_rf(X_df, y_df, n_estimators=n)

    # ========================================================
    # Step 3: PDP for Hot-dry (CHMD)
    # ========================================================
    print("\n" + "=" * 60)
    print("Step 3: Computing PDP for hot-dry (CHMD)...")
    print("=" * 60)

    for var in features:
        pdp = partial_dependence(
            rf_chmd,
            X_df,
            [var],
            grid_resolution=50
        )
        values = pdp['grid_values'][0]
        avg = pdp['average'][0]
        df_pdp = pd.DataFrame({
            'X': values,
            'Y': avg
        })
        df_pdp.to_csv(os.path.join(RF_OUTPUT_DIR, 'CHFD_rf', f'{var}.csv'))

    # Partial dependence display
    features_to_plot = features
    PartialDependenceDisplay.from_estimator(
        rf_chmd,
        X_df,
        features_to_plot,
        grid_resolution=50
    )
    plt.show()

    # ========================================================
    # Step 4: Feature Importance for Hot-dry (CHMD)
    # ========================================================
    print("\n" + "=" * 60)
    print("Step 4: Computing feature importance for hot-dry (CHMD)...")
    print("=" * 60)

    importances = rf_chmd.feature_importances_

    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print(importance_df)
    importance_df.to_csv(os.path.join(RF_OUTPUT_DIR, 'CHFD_rf', 'CHFD_importance.csv'), index=False)

    # ========================================================
    # Step 5: Train RF for Dry (NHMD) events
    # ========================================================
    print("\n" + "=" * 60)
    print("Step 5: Training RF for dry (NHMD) events...")
    print("=" * 60)

    y_nh = cold_grid_ds.rec_sm_min.where(mask)

    X_nh = xr.Dataset({
        'p_anom': cold_result.p.where(mask),
        'sshf_anom': cold_result.sshf.where(mask),
        'slhf_anom': cold_result.slhf.where(mask),
        'ssr_ano': cold_result.ssr.where(mask),
        'rh_anom': cold_result.rh.where(mask),
        'tmax_anom': cold_result.t.where(mask),
        'p': pr_cli.where(mask),
        'p_cv': pr_cv.where(mask),
        'tmax': tmax_mean.where(mask),
        'tmax_var': tmax_var.where(mask),
        'ai': ai.ai.where(mask)})

    df_nh = xr.merge([X_nh, y_nh]).to_dataframe().dropna()

    X_df_nh = df_nh[features]
    y_df_nh = df_nh['rec_sm_min']

    rf_nhmd = train_rf(X_df_nh, y_df_nh, n_estimators=300)

    for n in [200, 300, 400, 500]:
        print(f"n_estimators = {n}")
        train_rf(X_df_nh, y_df_nh, n_estimators=n)

    # ========================================================
    # Step 6: PDP for Dry (NHMD)
    # ========================================================
    print("\n" + "=" * 60)
    print("Step 6: Computing PDP for dry (NHMD)...")
    print("=" * 60)

    for var in features:
        pdp = partial_dependence(
            rf_nhmd,
            X_df_nh,
            [var],
            grid_resolution=50
        )
        values = pdp['grid_values'][0]
        avg = pdp['average'][0]
        df_pdp = pd.DataFrame({
            'X': values,
            'Y': avg
        })
        df_pdp.to_csv(os.path.join(RF_OUTPUT_DIR, 'NHFD_rf', f'{var}.csv'))

    # ========================================================
    # Step 7: Feature Importance for Dry (NHMD)
    # ========================================================
    print("\n" + "=" * 60)
    print("Step 7: Computing feature importance for dry (NHMD)...")
    print("=" * 60)

    importances_nh = rf_nhmd.feature_importances_

    importance_df_nh = pd.DataFrame({
        'feature': features,
        'importance': importances_nh
    }).sort_values(by='importance', ascending=False)

    print(importance_df_nh)
    importance_df_nh.to_csv(os.path.join(RF_OUTPUT_DIR, 'NHFD_rf', 'NHFD_importance.csv'), index=False)

    # ========================================================
    # Step 8: Importance Comparison Plot (Figure 16)
    # ========================================================
    print("\n" + "=" * 60)
    print("Step 8: Generating importance comparison plot...")
    print("=" * 60)

    chfd = pd.read_csv(os.path.join(RF_OUTPUT_DIR, 'CHFD_rf', 'CHFD_importance.csv'))
    nhfd = pd.read_csv(os.path.join(RF_OUTPUT_DIR, 'NHFD_rf', 'NHFD_importance.csv'))

    plt.rcParams['xtick.labelsize'] = 14

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
    plt.show()

    fig.savefig(os.path.join(OUTPUT_DIR, 'figure16.png'), dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print("\n" + "=" * 60)
    print("RF analysis complete!")
    print("=" * 60)

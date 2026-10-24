import xarray as xr
import numpy as np
import pandas as pd
from cftime import num2date
import cftime
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.decomposition import PCA

def get_weight(lon_val, lat_val):
    for (lon_min, lon_max, lat_min, lat_max), weight in weight_dict.items():
        lon_val_360 = lon_val % 360
        lon_min_360 = lon_min % 360
        lon_max_360 = lon_max % 360

        if lon_min_360 > lon_max_360:
            if (lon_val_360 >= lon_min_360) or (lon_val_360 <= lon_max_360):
                lat_match = (lat_min <= lat_val <= lat_max)
                if lat_match:
                    return weight / 100.0
        else:
            if (lon_min_360 <= lon_val_360 <= lon_max_360) and (lat_min <= lat_val <= lat_max):
                return weight / 100.0
    return 0.0


if __name__ == '__main__':

    df = pd.read_csv("F:/aoaoproject/ocean_pacific_percent.csv")
    df["lon_center"] = (df["lon_min"] + df["lon_max"]) / 2
    df["lat_center"] = (df["lat_min"] + df["lat_max"]) / 2

    weight_dict = {
        (row.lon_min, row.lon_max, row.lat_min, row.lat_max): row.ocean_percent
        for _, row in df.iterrows()
    }

    sst = xr.open_dataset('F:/guance_data/JRA-3Q/combined_sst.nc')
    sst = sst.isel(time=slice(376, 933))
    sst = sst.rename({"wtmp-sfc-fc-ll125-mn": "sst"})

    msl = xr.open_dataset('F:/guance_data/JRA-3Q/combined_msl.nc')
    msl = msl.isel(time=slice(376, 933))
    msl = msl.rename({"prmsl-msl-an-ll125-mn": "msl"})

    u10 = xr.open_dataset('F:/guance_data/JRA-3Q/combined_u10.nc')
    u10 = u10.isel(time=slice(376, 933))
    u10 = u10.rename({"ugrd10m-hgt-an-ll125-mn": "u10"})

    v10 = xr.open_dataset('F:/guance_data/JRA-3Q/combined_v10.nc')
    v10 = v10.isel(time=slice(376, 933))
    v10 = v10.rename({"vgrd10m-hgt-an-ll125-mn": "v10"})

    olr = xr.open_dataset('F:/guance_data/JRA-3Q/olr-monthly_197901_202505.nc')
    olr = olr.resample(time='MS').mean()

    datasets = {
        'sst': sst,  # 'pt' 对应的 Dataset
        'msl': msl,  # 'msl' 对应的 Dataset
        'olr': olr,  # 'olr' 对应的 Dataset
        'u10': u10,  # 'u10' 对应的 Dataset
        'v10': v10  # 'v10' 对应的 Dataset
    }

    variables = ["sst", "msl", "olr", "u10", "v10"]

    vectorized_weight = np.vectorize(get_weight)

    combined_data = []
    for var in variables:
        ds_var = datasets[var]

        lat_cond = (ds_var.lat >= -30) & (ds_var.lat <= 30)
        lat_cond_computed = lat_cond.compute()
        lon_cond = (ds_var.lon >= 100) & (ds_var.lon <= 290)
        lon_cond_computed = lon_cond.compute()
        region_mask = lat_cond_computed & lon_cond_computed
        ds_var = ds_var.where(region_mask, drop=True)

        ds_var["lon"] = (ds_var.lon + 180) % 360 - 180
        lon_grid, lat_grid = np.meshgrid(ds_var.lon.values, ds_var.lat.values)
        mask_ocean = ~np.isnan(ds_var.isel(time=0))
        ds_var = ds_var.where(mask_ocean)

        clim_var = ds_var.sel(time=slice('1981-01-01', '2018-12-31'))
        clim_monthly_var = clim_var.groupby('time.month').mean('time')
        def subtract_climatology(data):
            month = data['time.month']
            return data - clim_monthly_var.sel(month=month)
        anomaly_var = ds_var.groupby('time.month').map(subtract_climatology)

        std_dev_var = anomaly_var.std(dim='time', keep_attrs=True)
        anomaly_var = anomaly_var / std_dev_var

        weight_matrix = vectorized_weight(lon_grid, lat_grid)

        weight_da = xr.DataArray(
            weight_matrix,
            dims=["lat", "lon"],
            coords={"lat": ds_var.lat, "lon": ds_var.lon},
            name="weight"
        )

        weight_expanded = weight_da.expand_dims(time=ds_var.time)
        anomaly_var[var] = anomaly_var[var] * weight_expanded
        anomaly_var = anomaly_var.dropna(dim="lat", how="all").dropna(dim="lon", how="all")
        anomaly_var["lon"] = (anomaly_var.lon + 360) % 360
        anomaly_var = anomaly_var.sortby("lon")

        lat_radians = np.deg2rad(anomaly_var['lat'])
        cos_lat_weight = np.cos(lat_radians)
        anomaly_var[var] = anomaly_var[var] * cos_lat_weight

        ds_var_sliding = anomaly_var.rolling(time=2, min_periods=1).mean()

        var_data = ds_var_sliding[var].stack(ncol=["lat", "lon"]).transpose("time", "ncol")
        var_data_expanded = var_data.expand_dims(variable=[var])
        combined_data.append(var_data_expanded)


    ds_sst = combined_data[0]
    ds_ulwrf = combined_data[4]
    ds_slp = combined_data[1]
    ds_uwnd = combined_data[2]
    ds_vwnd = combined_data[3]

    combined_ds = xr.concat([ds_sst, ds_slp, ds_uwnd, ds_vwnd, ds_ulwrf], dim="variable")
    combined_unstacked = combined_ds.unstack("ncol")
    combined_stacked = combined_unstacked.stack(feature=("variable", "lat", "lon")).transpose("time", "feature")
    data = combined_stacked.values

    imputer = SimpleImputer(strategy="constant",fill_value=0)
    data_filled = imputer.fit_transform(data)

    pca_model = PCA(n_components=1)
    X_pca = pca_model.fit_transform(data_filled)
    pca = X_pca[:, 0]
    MEIv2 = (pca - pca.mean()) / pca.std()

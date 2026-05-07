import numpy as np
import pandas as pd


def calculate_trajectory_minimum_curvature(survey_df):
    df = survey_df.copy()

    df = df.rename(
        columns={
            "MD": "MD",
            "Measured Depth": "MD",
            "Inclination": "Inclination",
            "Inc": "Inclination",
            "Azimuth": "Azimuth",
            "Azi": "Azimuth",
        }
    )

    required = ["MD", "Inclination", "Azimuth"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[required].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    df = df.sort_values("MD").reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("At least two survey stations are required.")

    tvd = [0.0]
    northing = [0.0]
    easting = [0.0]
    dls_list = [0.0]

    for i in range(1, len(df)):
        md1 = df.loc[i - 1, "MD"]
        md2 = df.loc[i, "MD"]

        inc1 = np.radians(df.loc[i - 1, "Inclination"])
        inc2 = np.radians(df.loc[i, "Inclination"])

        azi1 = np.radians(df.loc[i - 1, "Azimuth"])
        azi2 = np.radians(df.loc[i, "Azimuth"])

        delta_md = md2 - md1

        if delta_md <= 0:
            raise ValueError("MD values must increase from top to bottom.")

        dogleg = np.arccos(
            np.cos(inc2 - inc1)
            - np.sin(inc1) * np.sin(inc2) * (1 - np.cos(azi2 - azi1))
        )

        if abs(dogleg) < 1e-12:
            rf = 1.0
        else:
            rf = (2 / dogleg) * np.tan(dogleg / 2)

        delta_tvd = delta_md / 2 * (np.cos(inc1) + np.cos(inc2)) * rf

        delta_north = (
            delta_md
            / 2
            * (
                np.sin(inc1) * np.cos(azi1)
                + np.sin(inc2) * np.cos(azi2)
            )
            * rf
        )

        delta_east = (
            delta_md
            / 2
            * (
                np.sin(inc1) * np.sin(azi1)
                + np.sin(inc2) * np.sin(azi2)
            )
            * rf
        )

        tvd.append(tvd[-1] + delta_tvd)
        northing.append(northing[-1] + delta_north)
        easting.append(easting[-1] + delta_east)

        dls = np.degrees(dogleg) * 100 / delta_md
        dls_list.append(dls)

    df["TVD"] = tvd
    df["Northing"] = northing
    df["Easting"] = easting
    df["DLS"] = dls_list

    return df


def interpolate_at_md(trajectory_df, md_value):
    if md_value is None:
        return None

    if md_value < trajectory_df["MD"].min() or md_value > trajectory_df["MD"].max():
        return None

    result = {}

    for col in ["TVD", "Northing", "Easting", "Inclination", "Azimuth"]:
        result[col] = np.interp(
            md_value,
            trajectory_df["MD"],
            trajectory_df[col],
        )

    result["MD"] = md_value

    return result
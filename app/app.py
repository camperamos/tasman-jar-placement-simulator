import sys
import base64
from pathlib import Path
from io import StringIO

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from calculations.survey_trajectory import (
    calculate_trajectory_minimum_curvature,
    interpolate_at_md,
)


FT_TO_M = 0.3048
M_TO_FT = 3.28084
TASMAN_BLUE = "#0079A8"
TASMAN_ORANGE = "#F47B20"
TASMAN_DARK = "#26323F"
DISCLAIMER_TEXT = (
    "Disclaimer: Results are based on theoretical calculations and simplified engineering assumptions. "
    "They are provided for planning support only. All rights reserved."
)
LOGO_PATH = ROOT_DIR / "assets" / "tasman_logo.png"

_TUBING_CSV = """OD,WT,ID
1.050,1.14,0.824
1.050,1.20,0.824
1.050,1.50,0.742
1.315,1.70,1.049
1.315,1.80,1.049
1.315,2.25,0.957
1.660,2.10,1.410
1.660,2.30,1.380
1.660,2.40,1.380
1.660,3.02,1.278
1.900,2.40,1.650
1.900,2.75,1.610
1.900,2.90,1.610
1.900,3.64,1.500
2.063,3.25,1.751
2.375,4.00,2.041
2.375,4.10,2.041
2.375,4.60,1.995
2.375,4.70,1.995
2.375,5.00,1.947
2.375,5.30,1.939
2.375,5.80,1.867
2.375,5.95,1.867
2.375,6.20,1.853
2.375,7.70,1.703
2.875,5.90,2.469
2.875,6.40,2.441
2.875,6.50,2.441
2.875,7.90,2.323
2.875,8.60,2.259
2.875,8.70,2.259
2.875,9.50,2.195
2.875,10.70,2.091
2.875,11.00,2.065
3.500,7.70,3.068
3.500,8.50,3.018
3.500,9.20,2.992
3.500,9.30,2.992
3.500,10.20,2.797
3.500,12.70,2.750
3.500,12.80,2.764
3.500,12.95,2.750
3.500,14.90,2.602
3.500,15.80,2.548
3.500,16.70,2.480
4.000,9.25,3.548
4.000,9.50,3.548
4.000,11.00,3.476
4.000,11.60,3.428
4.000,13.40,3.340
4.500,11.00,4.026
4.500,11.60,3.428
4.500,11.80,3.990
4.500,12.60,3.958
4.500,12.75,3.958
4.500,13.50,3.920
4.500,15.50,3.826
4.500,19.20,3.640
"""

_DRILLPIPE_CSV = """OD,WT,ID
2-3/8,4.80,2.000
2-3/8,4.85,1.995
2-3/8,6.65,1.815
2-7/8,6.45,2.469
2-7/8,6.85,2.441
2-7/8,8.35,2.323
2-7/8,10.40,2.151
3-1/2,8.50,3.063
3-1/2,9.50,2.992
3-1/2,11.20,2.900
3-1/2,13.30,2.764
3-1/2,15.50,2.602
4,10.40,3.500
4,11.85,3.476
4,12.50,3.382
4,14.00,3.340
4,15.70,3.240
4-1/2,12.75,4.000
4-1/2,13.75,3.958
4-1/2,16.60,3.826
4-1/2,18.10,3.754
4-1/2,20.00,3.640
5,16.25,4.408
5,19.50,4.276
5,25.60,4.000
5-1/2,21.90,4.778
5-1/2,24.70,4.670
5-9/16,19.00,4.975
5-9/16,22.20,4.859
5-9/16,25.25,4.733
6-5/8,22.20,6.065
6-5/8,25.20,5.965
6-5/8,31.90,5.761
7-5/8,29.25,6.969
8-5/8,40.00,7.825
"""

_CT_CSV = """OD,WT,ID
1.25,0.109,1.032
1.25,0.116,1.018
1.25,0.125,1.000
1.25,0.134,0.982
1.25,0.145,0.960
1.25,0.156,0.938
1.25,0.175,0.900
1.25,0.190,0.870
1.25,0.204,0.842
1.50,0.109,1.282
1.50,0.116,1.268
1.50,0.125,1.250
1.50,0.134,1.232
1.50,0.145,1.210
1.50,0.156,1.188
1.50,0.175,1.150
1.50,0.190,1.120
1.50,0.204,1.092
1.50,0.224,1.052
1.75,0.109,1.532
1.75,0.116,1.518
1.75,0.125,1.500
1.75,0.134,1.482
1.75,0.145,1.460
1.75,0.156,1.438
1.75,0.175,1.400
1.75,0.204,1.342
1.75,0.224,1.302
1.75,0.236,1.278
1.75,0.250,1.250
2.00,0.109,1.782
2.00,0.116,1.768
2.00,0.125,1.750
2.00,0.134,1.732
2.00,0.145,1.710
2.00,0.156,1.688
2.00,0.175,1.650
2.00,0.190,1.620
2.00,0.204,1.592
2.00,0.224,1.552
2.00,0.236,1.528
2.00,0.250,1.500
2.00,0.276,1.448
2.375,0.125,2.125
2.375,0.134,2.107
2.375,0.145,2.085
2.375,0.156,2.063
2.375,0.175,2.025
2.375,0.190,1.995
2.375,0.204,1.967
2.375,0.224,1.927
2.375,0.236,1.903
2.375,0.250,1.875
2.375,0.276,1.823
2.625,0.134,2.357
2.625,0.145,2.335
2.625,0.156,2.313
2.625,0.175,2.275
2.625,0.190,2.245
2.625,0.204,2.217
2.625,0.224,2.177
2.625,0.236,2.153
2.625,0.250,2.125
2.625,0.276,2.073
2.875,0.145,2.585
2.875,0.156,2.563
2.875,0.175,2.525
2.875,0.190,2.495
2.875,0.204,2.467
2.875,0.224,2.427
2.875,0.236,2.403
2.875,0.250,2.375
2.875,0.276,2.323
"""

TUBING_DF = pd.read_csv(StringIO(_TUBING_CSV))
DRILLPIPE_DF = pd.read_csv(StringIO(_DRILLPIPE_CSV))
CT_DF = pd.read_csv(StringIO(_CT_CSV))

_CASING_CSV = """OD,WT,ID
4,5.65,3.607
4,9.50,3.500
4,11.60,3.428
4-1/2,6.75,4.216
4-1/2,9.50,4.090
4-1/2,10.50,4.052
4-1/2,11.00,4.026
4-1/2,11.60,4.000
4-1/2,12.60,3.958
4-1/2,13.50,3.920
4-1/2,15.10,3.826
4-1/2,16.60,3.754
4-1/2,18.80,3.640
4-3/4,9.50,4.364
4-3/4,16.00,4.082
4-3/4,18.00,4.000
5,8.00,4.696
5,11.50,4.560
5,13.00,4.494
5,15.00,4.408
5,18.00,4.276
5,20.30,4.184
5,20.80,4.156
5,21.00,4.154
5,23.20,4.044
5,24.10,4.000
5-1/4,8.50,4.944
5-1/4,10.00,4.886
5-1/4,13.00,4.768
5-1/4,16.00,4.648
5-1/2,9.00,5.192
5-1/2,13.00,5.044
5-1/2,14.00,5.012
5-1/2,15.00,4.974
5-1/2,15.50,4.950
5-1/2,17.00,4.892
5-1/2,20.00,4.778
5-1/2,23.00,4.670
5-1/2,25.00,4.580
5-1/2,26.00,4.548
5-3/4,14.00,5.290
5-3/4,17.00,5.190
5-3/4,19.50,5.090
5-3/4,22.50,4.990
5-3/4,25.20,4.890
6,10.50,5.672
6,12.00,5.620
6,15.00,5.524
6,16.00,5.500
6,17.00,5.450
6,18.00,5.424
6,20.00,5.352
6,23.00,5.000
6,26.00,5.140
6-5/8,12.00,6.287
6-5/8,13.00,6.255
6-5/8,17.00,6.135
6-5/8,20.00,6.049
6-5/8,22.00,5.989
6-5/8,24.00,5.921
6-5/8,26.00,5.855
6-5/8,28.00,5.791
6-5/8,29.00,5.761
6-5/8,32.00,5.675
6-5/8,34.00,5.595
7,13.00,6.520
7,17.00,6.538
7,20.00,6.456
7,22.00,6.398
7,23.00,6.366
7,24.00,6.336
7,26.00,6.276
7,28.00,6.214
7,29.00,6.184
7,30.00,6.154
7,32.00,6.094
7,33.70,6.048
7,35.00,6.004
7,38.00,5.920
7,40.00,5.836
7-5/8,14.75,7.263
7-5/8,20.00,7.125
7-5/8,24.00,7.025
7-5/8,26.40,6.969
7-5/8,29.70,6.875
7-5/8,33.70,6.765
7-5/8,39.00,6.625
7-5/8,45.00,6.445
7-5/8,45.30,6.435
8,16.00,7.628
8,20.00,7.528
8,26.00,7.386
8-1/8,28.00,7.485
8-1/8,32.00,7.385
8-1/8,35.50,7.285
8-1/8,39.50,7.185
8-1/8,42.00,7.125
8-5/8,20.00,8.191
8-5/8,24.00,8.097
8-5/8,28.00,8.017
8-5/8,32.00,7.921
8-5/8,36.00,7.825
8-5/8,38.00,7.775
8-5/8,40.00,7.725
8-5/8,43.00,7.651
8-5/8,44.00,7.625
8-5/8,48.00,7.537
8-5/8,49.00,7.511
9,34.00,8.290
9,38.00,8.196
9,40.00,8.150
9,45.00,8.032
9,50.20,7.910
9,55.00,7.812
9-5/8,29.30,9.063
9-5/8,32.30,9.001
9-5/8,36.00,8.921
9-5/8,40.00,8.835
9-5/8,43.60,8.775
9-5/8,47.00,8.681
9-5/8,53.50,8.535
9-5/8,58.40,8.435
9-5/8,61.10,8.375
9-5/8,71.80,8.125
10,33.00,9.384
10,41.50,9.200
10,45.50,9.120
10,50.50,9.016
10,55.50,8.908
10,61.20,8.690
10-3/4,32.75,10.192
10-3/4,35.75,10.136
10-3/4,40.50,10.050
10-3/4,45.50,9.950
10-3/4,51.00,9.850
10-3/4,54.00,9.784
10-3/4,55.00,9.760
10-3/4,60.70,9.660
10-3/4,65.70,9.560
10-3/4,71.10,9.450
11,26.75,10.552
11-3/4,38.00,11.150
11-3/4,42.00,11.084
11-3/4,47.00,11.000
11-3/4,54.00,10.880
11-3/4,60.00,10.772
11-3/4,65.00,10.682
12,31.50,11.514
12,40.00,11.384
12-3/4,43.00,12.130
12-3/4,53.00,11.970
13,36.50,12.482
13,40.00,12.438
13,45.00,12.360
13,50.00,12.282
13,54.00,12.220
13-3/8,48.00,12.715
13-3/8,54.50,12.615
13-3/8,61.00,12.515
13-3/8,68.00,12.415
13-3/8,72.00,12.347
13-3/8,77.00,12.275
13-3/8,83.00,12.175
13-3/8,85.00,12.159
13-3/8,92.00,12.031
13-3/8,98.00,11.937
14,42.00,13.488
14,50.00,13.344
15,47.50,14.418
16,52.50,15.396
16,55.00,15.375
16,65.00,15.250
16,70.00,15.198
16,75.00,15.125
16,84.00,15.010
16,109.00,14.688
18-5/8,78.00,17.855
18-5/8,87.50,17.755
18-5/8,96.50,17.655
20,90.00,19.166
20,94.00,19.124
20,106.50,19.000
20,133.00,18.730
21-1/2,92.50,20.710
21-1/2,103.00,20.610
21-1/2,114.00,20.510
"""

CASING_DF = pd.read_csv(StringIO(_CASING_CSV))


def to_ft(value, unit):
    if value is None:
        return None
    return value * M_TO_FT if unit == "m" else value


def from_ft(value, unit):
    if value is None:
        return None
    return value * FT_TO_M if unit == "m" else value


def steel_weight_lbft(od_in, id_in):
    if od_in is None or id_in is None:
        return 0.0
    if od_in <= id_in:
        return 0.0
    return 2.67 * (od_in**2 - id_in**2)


def decimal_to_fraction_text(value):
    if value is None or pd.isna(value):
        return ""

    whole = int(np.floor(float(value)))
    frac = float(value) - whole
    common_fractions = {
        0.0: "",
        0.0625: "1/16",
        0.125: "1/8",
        0.1875: "3/16",
        0.25: "1/4",
        0.3125: "5/16",
        0.375: "3/8",
        0.4375: "7/16",
        0.5: "1/2",
        0.5625: "9/16",
        0.625: "5/8",
        0.6875: "11/16",
        0.75: "3/4",
        0.8125: "13/16",
        0.875: "7/8",
        0.9375: "15/16",
    }

    nearest = min(common_fractions, key=lambda x: abs(x - frac))
    if abs(nearest - frac) > 0.001:
        return f"{value:g}"

    fraction = common_fractions[nearest]
    if whole == 0:
        return fraction or "0"
    return f"{whole}-{fraction}" if fraction else str(whole)


def parse_inch_value(value):
    if value is None or pd.isna(value):
        return 0.0

    text = str(value).strip()
    if not text:
        return 0.0

    if "-" in text:
        whole_text, fraction_text = text.split("-", 1)
        whole = float(whole_text)
        numerator, denominator = fraction_text.split("/", 1)
        return whole + float(numerator) / float(denominator)

    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)

    return float(text)


def work_string_catalog(kind):
    if kind == "Drill Pipe":
        catalog = DRILLPIPE_DF.copy()
    elif kind == "Tubing":
        catalog = TUBING_DF.copy()
    else:
        catalog = CT_DF.copy()

    catalog["OD_numeric"] = catalog["OD"].apply(parse_inch_value)
    catalog["OD_label"] = catalog["OD"].astype(str)
    if kind in ["Tubing", "Coiled Tubing"]:
        catalog["OD_label"] = catalog["OD_numeric"].apply(lambda x: f'{x:g}"')
    else:
        catalog["OD_label"] = catalog["OD_label"].apply(lambda x: f'{x}"')
    catalog["WT"] = pd.to_numeric(catalog["WT"], errors="coerce")
    catalog["ID"] = pd.to_numeric(catalog["ID"], errors="coerce")
    if kind == "Coiled Tubing":
        catalog["Weight_lbft"] = catalog.apply(
            lambda row: steel_weight_lbft(row["OD_numeric"], row["ID"]),
            axis=1,
        )
    else:
        catalog["Weight_lbft"] = catalog["WT"]
    catalog["WT_label"] = catalog["WT"].apply(lambda value: f"{value:g}")
    return catalog


def casing_catalog():
    catalog = CASING_DF.copy()
    catalog["OD_numeric"] = catalog["OD"].apply(parse_inch_value)
    catalog["OD_label"] = catalog["OD"].astype(str).apply(lambda x: f'{x}"')
    catalog["WT"] = pd.to_numeric(catalog["WT"], errors="coerce")
    catalog["ID"] = pd.to_numeric(catalog["ID"], errors="coerce")
    return catalog


def jar_activation_limit_lbf(tool, direction="up"):
    if tool is None:
        return None

    if direction == "down":
        keys = ["field_max_push_load_lbs", "jar_standard_push_test_lbs"]
    else:
        keys = ["field_max_pull_load_lbs", "jar_standard_pull_test_lbs"]

    for key in keys:
        value = tool.get(key)
        if value is not None and not pd.isna(value) and float(value) > 0:
            return float(value)

    return None


def build_tool_label(row):
    od = decimal_to_fraction_text(row.get("od_in"))
    tool_id = decimal_to_fraction_text(row.get("id_in"))
    connection = row.get("connection")
    if row.get("tool_type") == "Accelerator":
        product = "Energizer"
    else:
        product = row.get("product") or "Jar"

    size = f'{od}" OD'
    if tool_id:
        size = f'{od}" OD x {tool_id}" ID'

    return f"{size} | {connection} | {product}"


def ct_jar_catalog_rows():
    base = {
        "tool_type": "Jar",
        "application": "CT",
        "manufacturer": "Logan / Innovex",
        "product": "Bi-Directional Coiled Tubing Jar",
        "drill_collar_weight_range_lbs": "",
        "source": "Bi-Directional Coiled Tubing Jar Manual D410 p11/p14",
    }
    return [
        {
            **base,
            "assembly": "616-169",
            "display_name": "1-11/16 in Bi-Directional CT Jar",
            "length_ft": 4.2760,
            "od_in": 1.6875,
            "id_in": 0.530,
            "connection": '1" AMMT',
            "stroke_in": np.nan,
            "pump_open_area_sq_in": 1.28,
            "jar_standard_pull_test_lbs": 12000,
            "field_max_pull_load_lbs": 12000,
            "jar_standard_push_test_lbs": 1500,
            "field_max_push_load_lbs": 12000,
            "max_lift_load_lbs": 69000,
            "torque_yield_ft_lbs": 800,
        },
        {
            **base,
            "assembly": "616-213",
            "display_name": "2-1/8 in Bi-Directional CT Jar",
            "length_ft": 5.7656,
            "od_in": 2.125,
            "id_in": 0.656,
            "connection": '1-1/2" AMMT',
            "stroke_in": np.nan,
            "pump_open_area_sq_in": 2.23,
            "jar_standard_pull_test_lbs": 24000,
            "field_max_pull_load_lbs": 18000,
            "jar_standard_push_test_lbs": 2500,
            "field_max_push_load_lbs": 18000,
            "max_lift_load_lbs": 125000,
            "torque_yield_ft_lbs": 1700,
        },
        {
            **base,
            "assembly": "616-288",
            "display_name": "2-7/8 in Bi-Directional CT Jar",
            "length_ft": 6.0573,
            "od_in": 2.875,
            "id_in": 0.906,
            "connection": '2-3/8" PAC',
            "stroke_in": np.nan,
            "pump_open_area_sq_in": 3.98,
            "jar_standard_pull_test_lbs": 35000,
            "field_max_pull_load_lbs": 36000,
            "jar_standard_push_test_lbs": 4000,
            "field_max_push_load_lbs": 36000,
            "max_lift_load_lbs": 225000,
            "torque_yield_ft_lbs": 4000,
        },
    ]


def ct_energizer_catalog_rows():
    base = {
        "tool_type": "Accelerator",
        "application": "CT",
        "manufacturer": "Logan / Innovex",
        "product": "Bi-Directional Coiled Tubing Energizer",
        "drill_collar_weight_range_lbs": "",
        "source": "Bi-Directional Coiled Tubing Energizers Catalog",
    }
    return [
        {
            **base,
            "assembly": "617-169",
            "display_name": "1-11/16 in Bi-Directional CT Energizer",
            "length_ft": 4.2760,
            "od_in": 1.6875,
            "id_in": 0.530,
            "connection": '1" AMMT',
            "stroke_in": 3.95,
            "pump_open_area_sq_in": 1.128,
            "field_max_pull_load_lbs": 12000,
            "field_max_push_load_lbs": 12000,
            "max_lift_load_lbs": 69000,
            "tensile_yield_lbs": 69000,
            "torque_yield_ft_lbs": 800,
        },
        {
            **base,
            "assembly": "617-213",
            "display_name": "2-1/8 in Bi-Directional CT Energizer",
            "length_ft": 5.7656,
            "od_in": 2.125,
            "id_in": 0.656,
            "connection": '1-1/2" AMMT',
            "stroke_in": 5.50,
            "pump_open_area_sq_in": 2.23,
            "field_max_pull_load_lbs": 18000,
            "field_max_push_load_lbs": 18000,
            "max_lift_load_lbs": 125000,
            "tensile_yield_lbs": 125000,
            "torque_yield_ft_lbs": 1700,
        },
        {
            **base,
            "assembly": "617-288",
            "display_name": "2-7/8 in Bi-Directional CT Energizer",
            "length_ft": 6.0573,
            "od_in": 2.875,
            "id_in": 0.906,
            "connection": '2-3/8" PAC',
            "stroke_in": 6.00,
            "pump_open_area_sq_in": 3.98,
            "field_max_pull_load_lbs": 36000,
            "field_max_push_load_lbs": 36000,
            "max_lift_load_lbs": 225000,
            "tensile_yield_lbs": 225000,
            "torque_yield_ft_lbs": 4000,
        },
    ]


@st.cache_data
def load_tool_catalog():
    catalog_path = ROOT_DIR / "data" / "logan_tools.csv"
    if not catalog_path.exists():
        return pd.DataFrame()

    catalog = pd.read_csv(catalog_path)
    catalog["application"] = "Fishing"
    catalog = pd.concat(
        [
            catalog,
            pd.DataFrame(ct_jar_catalog_rows()),
            pd.DataFrame(ct_energizer_catalog_rows()),
        ],
        ignore_index=True,
        sort=False,
    )

    numeric_cols = [
        "length_ft",
        "od_in",
        "id_in",
        "stroke_in",
        "pump_open_area_sq_in",
        "jar_standard_pull_test_lbs",
        "field_max_pull_load_lbs",
        "jar_standard_push_test_lbs",
        "field_max_push_load_lbs",
        "max_lift_load_lbs",
        "tensile_yield_lbs",
        "torque_yield_ft_lbs",
    ]

    for col in numeric_cols:
        if col in catalog.columns:
            catalog[col] = pd.to_numeric(catalog[col], errors="coerce")

    catalog["option_label"] = catalog.apply(build_tool_label, axis=1)

    return catalog


def tool_options(catalog, tool_type, application=None):
    if catalog.empty:
        return []

    tools = catalog[catalog["tool_type"] == tool_type].copy()
    if application is not None and "application" in tools.columns:
        tools = tools[tools["application"] == application].copy()
    return tools["option_label"].dropna().tolist()


def selected_tool(catalog, tool_type, option_label, application=None):
    if catalog.empty or not option_label:
        return None

    matches = catalog[
        (catalog["tool_type"] == tool_type)
        & (catalog["option_label"] == option_label)
    ]
    if application is not None and "application" in matches.columns:
        matches = matches[matches["application"] == application]

    if matches.empty:
        return None

    return matches.iloc[0]


def tool_length_for_unit(tool, unit):
    if tool is None or pd.isna(tool.get("length_ft")):
        return 0.0
    return from_ft(float(tool["length_ft"]), unit)


def tool_dimension(tool, key, fallback):
    if tool is None or pd.isna(tool.get(key)):
        return fallback
    return float(tool[key])


def dataframes_match(left, right):
    left_cmp = left.reset_index(drop=True).astype(object).where(pd.notna(left), None)
    right_cmp = right.reset_index(drop=True).astype(object).where(pd.notna(right), None)
    return left_cmp.equals(right_cmp)


def default_component_description(component):
    return "" if component == "Other" else component


def apply_bha_component_defaults(bha_df, unit, jar_tool, accelerator_tool):
    work = bha_df.copy()
    length_col = f"Length ({unit})"

    if "Item" in work.columns:
        work = work.drop(columns=["Item"])
    if length_col not in work.columns:
        work[length_col] = 0.0
    for col in ["Description", "Joints", "OD (in)", "ID (in)", "Type"]:
        if col not in work.columns:
            work[col] = None
    for col in ["Component", "Description", "Type"]:
        work[col] = work[col].astype("object")

    component_values = work["Component"].fillna("").astype(str).str.strip()
    work["Component"] = component_values.replace("", np.nan)
    work["Joints"] = pd.to_numeric(work["Joints"], errors="coerce")
    non_other_rows = (component_values != "Other") & (component_values != "")
    work.loc[non_other_rows, "Description"] = component_values[non_other_rows].apply(
        default_component_description
    )

    component_type_map = {
        "Mill/Bit": "Mill/Bit",
        "Overshot / Spear": "Fishing Tool",
        "CT Connector": "CT Connector",
        "Hydraulic Disconnect": "Hydraulic Disconnect",
        "Dual Flapper": "Dual Flapper",
        "Circulation Sub": "Circulation Sub",
        "Stabilizer": "Stabilizer",
        "Safety Joint": "Accessory",
        "Bumper Sub": "Bumper Sub",
        "Weight Bar": "Weight Bar",
        "Drill Collar": "DC",
        "HWDP": "HWDP",
        "Drill Pipe": "Drill Pipe",
        "Crossover": "Crossover",
        "Jar": "Jar",
        "Accelerator": "Accelerator",
        "Other": "Other",
    }

    for component, component_type in component_type_map.items():
        rows = component_values == component
        work.loc[rows, "Type"] = component_type

    joint_based_rows = component_values.isin(["Drill Collar", "HWDP", "Weight Bar"])
    joint_length = 30.0 if unit == "ft" else FT_TO_M * 30.0
    work.loc[joint_based_rows & work["Joints"].isna(), "Joints"] = 1
    work.loc[joint_based_rows, length_col] = (
        work.loc[joint_based_rows, "Joints"].fillna(0.0) * joint_length
    )
    work.loc[~joint_based_rows, "Joints"] = np.nan

    if jar_tool is not None:
        jar_rows = component_values.str.lower() == "jar"
        jar_length = tool_length_for_unit(jar_tool, unit)
        if jar_length:
            work.loc[jar_rows, length_col] = jar_length
        work.loc[jar_rows, "OD (in)"] = tool_dimension(jar_tool, "od_in", 0.0)
        work.loc[jar_rows, "ID (in)"] = tool_dimension(jar_tool, "id_in", 0.0)
        work.loc[jar_rows, "Type"] = "Jar"
        work.loc[jar_rows, "Description"] = "Jar"

    if accelerator_tool is not None:
        accelerator_rows = component_values.str.lower() == "accelerator"
        accelerator_length = tool_length_for_unit(accelerator_tool, unit)
        if accelerator_length:
            work.loc[accelerator_rows, length_col] = accelerator_length
        work.loc[accelerator_rows, "OD (in)"] = tool_dimension(
            accelerator_tool,
            "od_in",
            0.0,
        )
        work.loc[accelerator_rows, "ID (in)"] = tool_dimension(
            accelerator_tool,
            "id_in",
            0.0,
        )
        work.loc[accelerator_rows, "Type"] = "Accelerator"
        work.loc[accelerator_rows, "Description"] = "Accelerator"

    return work


def tool_capacity_display(tool):
    if tool is None:
        return {}

    fields = {
        "Assembly": tool.get("assembly"),
        "Connection": tool.get("connection"),
        "Stroke (in)": tool.get("stroke_in"),
        "DC weight range (lb)": tool.get("drill_collar_weight_range_lbs"),
        "Pump open area (sq in)": tool.get("pump_open_area_sq_in"),
        "Low test pull (lb)": tool.get("jar_low_test_pull_lbs"),
        "Standard / test pull (lb)": tool.get("jar_standard_pull_test_lbs"),
        "Field max pull (lb)": tool.get("field_max_pull_load_lbs"),
        "Standard / test push (lb)": tool.get("jar_standard_push_test_lbs"),
        "Field max push (lb)": tool.get("field_max_push_load_lbs"),
        "Max lift load (lb)": tool.get("max_lift_load_lbs"),
        "Tensile @ yield (lb)": tool.get("tensile_yield_lbs"),
        "Torque @ yield (ft-lb)": tool.get("torque_yield_ft_lbs"),
    }

    display = {}
    for label, value in fields.items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            display[label] = (
                f"{value:,.0f}"
                if float(value).is_integer()
                else f"{value:,.3f}".rstrip("0").rstrip(".")
            )
        else:
            display[label] = str(value)

    return display


def point_for_plot(point, unit):
    if point is None:
        return None
    factor = FT_TO_M if unit == "m" else 1.0
    return {
        "MD": point["MD"] * factor,
        "TVD": point["TVD"] * factor,
        "Northing": point["Northing"] * factor,
        "Easting": point["Easting"] * factor,
        "Inclination": point["Inclination"],
        "Azimuth": point["Azimuth"],
    }


def add_marker_3d(fig, point, name, color, unit):
    point = point_for_plot(point, unit)
    if point is None:
        return

    fig.add_trace(
        go.Scatter3d(
            x=[point["Easting"]],
            y=[point["Northing"]],
            z=[-point["TVD"]],
            mode="markers+text",
            marker=dict(size=7, color=color),
            text=[name],
            textposition="top center",
            name=f"{name} @ {point['MD']:,.1f} {unit}",
        )
    )


def add_marker_2d(fig, point, x_key, y_key, name, color, unit, negative_y=True):
    point = point_for_plot(point, unit)
    if point is None:
        return

    y_val = -point[y_key] if negative_y else point[y_key]

    fig.add_trace(
        go.Scatter(
            x=[point[x_key]],
            y=[y_val],
            mode="markers+text",
            marker=dict(size=9, color=color),
            text=[name],
            textposition="top center",
            showlegend=False,
        )
    )


def add_marker_departure_tvd(fig, point, name, color, unit):
    point = point_for_plot(point, unit)
    if point is None:
        return

    horizontal_departure = np.sqrt(point["Easting"] ** 2 + point["Northing"] ** 2)

    fig.add_trace(
        go.Scatter(
            x=[horizontal_departure],
            y=[-point["TVD"]],
            mode="markers+text",
            marker=dict(size=9, color=color),
            text=[name],
            textposition="top center",
            showlegend=False,
        )
    )


def pipe_area_in2(od_in, id_in):
    if od_in is None or id_in is None or od_in <= id_in:
        return 0.0
    return np.pi / 4.0 * (od_in**2 - id_in**2)


def pipe_moment_inertia_in4(od_in, id_in):
    if od_in is None or id_in is None or od_in <= id_in:
        return 0.0
    return np.pi / 64.0 * (od_in**4 - id_in**4)


def estimate_survey_efficiency(trajectory_df, friction_coefficient):
    if trajectory_df.empty:
        return 1.0, {}

    md_range = max(float(trajectory_df["MD"].max() - trajectory_df["MD"].min()), 1.0)
    tvd_range = max(float(trajectory_df["TVD"].max() - trajectory_df["TVD"].min()), 1.0)
    avg_sin_inc = float(np.sin(np.radians(trajectory_df["Inclination"])).mean())
    tortuosity = max(md_range / tvd_range - 1.0, 0.0)
    avg_dls = float(trajectory_df["DLS"].mean()) if "DLS" in trajectory_df else 0.0
    max_dls = float(trajectory_df["DLS"].max()) if "DLS" in trajectory_df else 0.0

    contact_index = avg_sin_inc + 0.7 * tortuosity + 0.025 * avg_dls + 0.01 * max_dls
    efficiency = float(np.exp(-friction_coefficient * contact_index))
    efficiency = min(max(efficiency, 0.30), 1.0)

    details = {
        "Average sin(inclination)": avg_sin_inc,
        "MD/TVD tortuosity": tortuosity,
        "Average DLS": avg_dls,
        "Max DLS": max_dls,
        "Trajectory contact index": contact_index,
        "Trajectory efficiency": efficiency,
    }
    return efficiency, details


def estimate_ct_setdown_capacity(
    trajectory_df,
    hole_id_in,
    ct_od_in,
    ct_id_in,
    ct_weight_lbft,
    mud_weight_ppg,
    friction_coefficient,
    steel_modulus_psi,
    local_weight_above_jar_lbf=0.0,
):
    if (
        trajectory_df.empty
        or hole_id_in is None
        or ct_od_in is None
        or ct_id_in is None
        or hole_id_in <= ct_od_in
        or ct_od_in <= ct_id_in
    ):
        return 0.0, {
            "CT buckling status": "Invalid or incomplete CT / hole geometry",
        }

    md_range = max(float(trajectory_df["MD"].max() - trajectory_df["MD"].min()), 1.0)
    tvd_range = max(float(trajectory_df["TVD"].max() - trajectory_df["TVD"].min()), 1.0)
    avg_sin_inc = float(np.sin(np.radians(trajectory_df["Inclination"])).mean())
    tortuosity = max(md_range / tvd_range - 1.0, 0.0)
    radial_clearance_in = max((hole_id_in - ct_od_in) / 2.0, 0.05)
    clearance_ratio = radial_clearance_in / max(hole_id_in, 0.1)

    inertia_in4 = pipe_moment_inertia_in4(ct_od_in, ct_id_in)
    steel_density_ppg = 65.5
    buoyancy_factor = min(max(1.0 - mud_weight_ppg / steel_density_ppg, 0.60), 1.0)
    effective_weight_lbin = max(ct_weight_lbft * buoyancy_factor / 12.0, 0.001)

    # Approximate constrained helical-buckling onset load for CT in a wellbore.
    # Larger hole/CT clearance lowers the force that can be transmitted downward.
    critical_buckling_lbf = 2.0 * np.sqrt(
        max(steel_modulus_psi * inertia_in4 * effective_weight_lbin / radial_clearance_in, 0.0)
    )

    contact_index = avg_sin_inc + 0.85 * tortuosity + 2.5 * clearance_ratio
    transmission_efficiency = float(np.exp(-friction_coefficient * contact_index))
    transmission_efficiency = min(max(transmission_efficiency, 0.05), 1.0)

    local_weight_efficiency = min(max(1.0 - 1.6 * avg_sin_inc, 0.25), 1.0)
    usable_local_weight_lbf = max(local_weight_above_jar_lbf, 0.0) * local_weight_efficiency
    submerged_ct_weight_lbf = ct_weight_lbft * buoyancy_factor * md_range
    string_weight_transfer_factor = 0.10 * max(1.0 - 1.25 * avg_sin_inc - 0.45 * tortuosity, 0.12)
    string_weight_transfer_factor *= float(
        np.exp(-friction_coefficient * (0.70 * avg_sin_inc + 0.35 * tortuosity))
    )
    transmitted_ct_weight_lbf = submerged_ct_weight_lbf * string_weight_transfer_factor
    available_setdown_lbf = (
        critical_buckling_lbf * transmission_efficiency
        + transmitted_ct_weight_lbf
        + usable_local_weight_lbf
    )

    details = {
        "Hole ID (in)": hole_id_in,
        "CT OD (in)": ct_od_in,
        "CT ID (in)": ct_id_in,
        "Radial clearance (in)": radial_clearance_in,
        "Clearance ratio": clearance_ratio,
        "CT moment of inertia (in^4)": inertia_in4,
        "Buoyancy factor": buoyancy_factor,
        "Effective CT weight (lb/ft)": ct_weight_lbft * buoyancy_factor,
        "Submerged CT weight in hole (lb)": submerged_ct_weight_lbf,
        "CT string weight transfer factor": string_weight_transfer_factor,
        "Transferred CT weight before buckling (lb)": transmitted_ct_weight_lbf,
        "Average sin(inclination)": avg_sin_inc,
        "MD/TVD tortuosity": tortuosity,
        "CT buckling contact index": contact_index,
        "CT setdown transmission efficiency": transmission_efficiency,
        "CT critical buckling load (lb)": critical_buckling_lbf,
        "Usable local weight above jar (lb)": usable_local_weight_lbf,
        "Buckling-limited setdown at jar (lb)": available_setdown_lbf,
    }
    return available_setdown_lbf, details


def estimate_clearance_efficiency(hole_id_in, work_string_od_in, bha_df):
    od_values = [work_string_od_in]
    if "OD (in)" in bha_df.columns:
        od_values += pd.to_numeric(bha_df["OD (in)"], errors="coerce").dropna().tolist()

    max_od = max([float(x) for x in od_values if x is not None], default=0.0)
    if hole_id_in is None or hole_id_in <= 0 or max_od <= 0:
        return 1.0, {"Max OD (in)": max_od, "Minimum radial clearance (in)": None}

    radial_clearance = (hole_id_in - max_od) / 2.0
    clearance_ratio = max(radial_clearance / hole_id_in, 0.0)
    efficiency = 0.35 + 0.65 * min(clearance_ratio / 0.12, 1.0)
    efficiency = min(max(efficiency, 0.20), 1.0)

    details = {
        "Hole ID (in)": hole_id_in,
        "Max OD (in)": max_od,
        "Minimum radial clearance (in)": radial_clearance,
        "Clearance efficiency": efficiency,
    }
    return efficiency, details


def physical_impact_result(
    pull_load_lbf,
    joints_above_jar,
    work_string_length_ft,
    work_string_od_in,
    work_string_id_in,
    work_string_weight_lbft,
    joint_length_ft,
    joint_weight_lbft,
    fish_weight_lbf,
    fish_length_ft,
    jar_stroke_in,
    accelerator_stroke_in,
    accelerator_weight_lbf,
    trajectory_efficiency,
    clearance_efficiency,
    fluid_efficiency,
    steel_modulus_psi,
    stopping_distance_in,
    effective_area_floor_in2=4.0,
    accelerator_energy_factor=0.5,
):
    area_in2 = pipe_area_in2(work_string_od_in, work_string_id_in)
    effective_area_in2 = max(area_in2, effective_area_floor_in2)
    string_stretch_in = 0.0
    string_energy_ftlbf = 0.0

    if effective_area_in2 > 0 and steel_modulus_psi > 0:
        string_stretch_in = (
            pull_load_lbf * work_string_length_ft * 12.0 / (effective_area_in2 * steel_modulus_psi)
        )
        string_energy_ftlbf = 0.5 * pull_load_lbf * string_stretch_in / 12.0

    accelerator_energy_ftlbf = 0.0
    if accelerator_stroke_in and accelerator_stroke_in > 0:
        accelerator_energy_ftlbf = (
            accelerator_energy_factor * pull_load_lbf * accelerator_stroke_in / 12.0
        )

    stored_energy_ftlbf = string_energy_ftlbf + accelerator_energy_ftlbf
    delivered_energy_ftlbf = (
        stored_energy_ftlbf
        * trajectory_efficiency
        * clearance_efficiency
        * fluid_efficiency
    )

    stop_ft = max(stopping_distance_in / 12.0, 0.01)
    impact_lbf = pull_load_lbf + delivered_energy_ftlbf / stop_ft

    collar_weight_lbf = joints_above_jar * joint_length_ft * joint_weight_lbft
    moving_weight_lbf = max(collar_weight_lbf + accelerator_weight_lbf, 1.0)
    resisted_weight_lbf = max(fish_weight_lbf + 0.15 * fish_length_ft * joint_weight_lbft, 1.0)
    reference_weight_lbf = max(6.0 * joint_length_ft * joint_weight_lbft, 1.0)
    mass_coupling = 0.55 + 0.35 * (1.0 - np.exp(-moving_weight_lbf / reference_weight_lbf))
    fish_resistance = min(max(resisted_weight_lbf / (resisted_weight_lbf + moving_weight_lbf), 0.25), 0.85)
    spacing_length_ft = joints_above_jar * joint_length_ft
    spacing_efficiency = np.exp(-spacing_length_ft / 1800.0)

    delivered_energy_ftlbf = (
        delivered_energy_ftlbf * mass_coupling * fish_resistance * spacing_efficiency
    )

    moving_mass_slugs = moving_weight_lbf / 32.174
    impulse_lbsec = np.sqrt(max(2.0 * moving_mass_slugs * delivered_energy_ftlbf, 0.0))
    impact_lbf = pull_load_lbf + delivered_energy_ftlbf / stop_ft
    impact_ratio = impact_lbf / pull_load_lbf if pull_load_lbf > 0 else 0.0

    return {
        "Impact (lb)": impact_lbf,
        "Impulse (lb-sec)": impulse_lbsec,
        "Up Impact Ratio": impact_ratio,
        "String Stretch (in)": string_stretch_in,
        "String Energy (ft-lbf)": string_energy_ftlbf,
        "Accelerator Energy (ft-lbf)": accelerator_energy_ftlbf,
        "Delivered Energy (ft-lbf)": delivered_energy_ftlbf,
        "Moving Weight (lb)": moving_weight_lbf,
        "Fish Resistance Weight (lb)": resisted_weight_lbf,
        "Mass Coupling": mass_coupling,
        "Fish Resistance Factor": fish_resistance,
        "Spacing Efficiency": spacing_efficiency,
    }


def calculate_impact_by_joints(
    firing_load_lbf,
    max_joints,
    physics_inputs,
):
    rows = []

    for joints in range(0, max_joints + 1):
        result = physical_impact_result(
            pull_load_lbf=firing_load_lbf,
            joints_above_jar=joints,
            **physics_inputs,
        )
        rows.append(
            {
                "Number of Joints above Jar": joints,
                **result,
            }
        )

    return pd.DataFrame(rows)


def calculate_impact_by_pull_load(
    pull_start_lbf,
    pull_end_lbf,
    pull_step_lbf,
    selected_joint_count,
    physics_inputs,
):
    if pull_step_lbf <= 0:
        pull_step_lbf = 10000.0
    if pull_start_lbf > pull_end_lbf:
        pull_start_lbf, pull_end_lbf = pull_end_lbf, pull_start_lbf

    pull_loads = np.arange(pull_end_lbf, pull_start_lbf - 0.1, -pull_step_lbf)
    rows = []

    for pull_load in pull_loads:
        result = physical_impact_result(
            pull_load_lbf=pull_load,
            joints_above_jar=selected_joint_count,
            **physics_inputs,
        )
        rows.append(
            {
                "Pull Load (lb)": pull_load,
                **result,
            }
        )

    return pd.DataFrame(rows)


def format_report_number(value, decimals=0):
    if value is None or pd.isna(value):
        return ""
    text = f"{value:,.{decimals}f}"
    if decimals > 0:
        text = text.rstrip("0").rstrip(".")
    return text


def format_display_dataframe(df, decimal_cols=None):
    decimal_cols = decimal_cols or {}
    display = df.copy()

    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            decimals = decimal_cols.get(col, 0)
            display[col] = display[col].apply(
                lambda x: "" if pd.isna(x) else format_report_number(x, decimals)
            )

    return display


def apply_integer_axis_format(fig):
    fig.update_xaxes(tickformat=",.0f", separatethousands=True)
    fig.update_yaxes(tickformat=",.0f", separatethousands=True)
    fig.update_layout(
        yaxis=dict(tickformat=",.0f", separatethousands=True),
        yaxis2=dict(tickformat=",.0f", separatethousands=True),
    )
    return fig


def set_trace_hovertemplates(fig, x_label, primary_label="Impact", secondary_label="Impulse"):
    if len(fig.data) > 0:
        fig.data[0].hovertemplate = (
            f"{x_label}: %{{x:,.0f}}<br>{primary_label}: %{{y:,.0f}}<extra></extra>"
        )
    if len(fig.data) > 1:
        fig.data[1].hovertemplate = (
            f"{x_label}: %{{x:,.0f}}<br>{secondary_label}: %{{y:,.0f}}<extra></extra>"
        )
    return fig


def compact_dataframe(df, height=180, decimal_cols=None, width_ratio=0.62):
    left, right = st.columns([width_ratio, 1.0 - width_ratio])
    left.dataframe(
        format_display_dataframe(df, decimal_cols),
        use_container_width=True,
        height=height,
    )
    return left


def file_data_uri(path, mime_type):
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def plotly_report_html(fig, title):
    report_fig = go.Figure(fig)
    report_fig.update_layout(
        autosize=False,
        width=1100,
        height=540,
        margin=dict(l=80, r=80, t=70, b=70),
    )
    chart_html = report_fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": False},
    )
    return f"<h2>{title}</h2><div class=\"chart-block\">{chart_html}</div>"


def dataframe_to_report_html(df, title, decimal_cols=None):
    decimal_cols = decimal_cols or {}
    display = df.copy()

    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            decimals = decimal_cols.get(col, 0)
            display[col] = display[col].apply(lambda x: format_report_number(x, decimals))

    return f"<h2>{title}</h2>{display.to_html(index=False, escape=False)}"


def build_impact_report_html(report_sections):
    logo_uri = file_data_uri(LOGO_PATH, "image/png")
    logo_html = f'<img class="logo" src="{logo_uri}" alt="Tasman logo">' if logo_uri else ""
    styles = """
    <style>
      body { font-family: Arial, sans-serif; color: #26323F; margin: 28px; }
      .header { display: flex; align-items: center; gap: 24px; border-bottom: 5px solid #F47B20; padding-bottom: 14px; }
      .logo { width: 420px; max-width: 46%; height: auto; }
      h1 { margin-bottom: 4px; color: #0079A8; }
      h2 { margin-top: 28px; border-bottom: 1px solid #ccd2d8; padding-bottom: 4px; color: #0079A8; }
      table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 12px; }
      th, td { border: 1px solid #d8dde3; padding: 6px 8px; text-align: right; }
      th:first-child, td:first-child { text-align: left; }
      th { background: #eef2f6; color: #26323F; }
      .meta { color: #5b6773; margin-bottom: 20px; }
      .note { font-size: 12px; color: #5b6773; margin-top: 20px; border-top: 2px solid #F47B20; padding-top: 10px; }
      .chart-block { margin-top: 8px; page-break-inside: avoid; width: 100%; min-height: 520px; }
      .client-block { width: 58%; margin-top: 18px; }
      .client-row { display: grid; grid-template-columns: 190px 1fr; border-bottom: 1px solid #d8dde3; padding: 7px 10px; }
      .client-label { font-weight: 700; color: #26323F; }
      @media print {
        body { margin: 12mm; }
        .chart-block { page-break-inside: avoid; min-height: 520px; }
      }
    </style>
    """
    return f"""
    <html>
      <head>{styles}<title>Jar Impact Simulation Report</title></head>
      <body>
        <div class="header">
          {logo_html}
          <div>
            <h1>Jar Impact Simulation Report</h1>
            <div class="meta">{report_sections["meta"]}</div>
          </div>
        </div>
        {report_sections["body"]}
        <div class="note">
          {DISCLAIMER_TEXT}
        </div>
      </body>
    </html>
    """


st.set_page_config(
    page_title="Tasman Jar Placement Simulator",
    layout="wide",
)

st.markdown(
    f"""
    <style>
      div.stButton > button,
      div.stDownloadButton > button {{
        background-color: {TASMAN_ORANGE};
        border-color: {TASMAN_ORANGE};
        color: #ffffff;
        font-weight: 700;
      }}
      div.stButton > button:hover,
      div.stDownloadButton > button:hover {{
        background-color: #d9650b;
        border-color: #d9650b;
        color: #ffffff;
      }}
      div.stButton > button:focus,
      div.stDownloadButton > button:focus {{
        box-shadow: 0 0 0 0.2rem rgba(244, 123, 32, 0.25);
        color: #ffffff;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([0.28, 0.72])
if LOGO_PATH.exists():
    header_left.image(str(LOGO_PATH), width=420)
header_right.title("Tasman Jar Placement Simulator")
header_right.caption("Survey, fish, BHA, work string and well trajectory")
header_right.caption(DISCLAIMER_TEXT)

st.markdown("---")

st.markdown("## Report Header")
rh1, rh2 = st.columns(2)
well_name = rh1.text_input("Well Name", value="")
customer = rh2.text_input("Customer", value="")

rh4, rh5 = st.columns(2)
customer_representative = rh4.text_input("Customer Rep", value="")
customer_email = rh5.text_input("Customer Rep Email", value="")

prepared_by = st.text_input("Prepared by", value="")

st.markdown("---")

unit = st.radio(
    "Depth / Length Unit",
    ["ft", "m"],
    horizontal=True,
    help="Select the unit used for survey MD, fish depth, fish component lengths and BHA component lengths.",
)

st.markdown("## Jar System")
jar_system = st.radio(
    "Select the jar family for this simulation",
    ["Fishing Jars", "CT Jars"],
    horizontal=True,
    help="Fishing Jars uses the fishing jar/energizer catalog. CT Jars uses bidirectional CT jars and CT energizers.",
)
catalog_application = "CT" if jar_system == "CT Jars" else "Fishing"
is_ct_job = jar_system == "CT Jars"
ct_operation_type = None
is_ct_milling_operation = False
if is_ct_job:
    ct_operation_type = st.radio(
        "CT BHA Type",
        ["Fishing", "Milling"],
        horizontal=True,
        help="Fishing shows TOF, stuck point and fish components. Milling assumes impact at the mill/bit and skips fish data.",
    )
    is_ct_milling_operation = ct_operation_type == "Milling"

st.markdown("## Well Profile")
well_profile = st.radio(
    "Well Trajectory",
    ["Vertical", "Deviated"],
    horizontal=True,
)

example = ""

survey_text = ""
well_depth_input = None

if well_profile == "Vertical":
    well_depth_input = st.number_input(
        f"Well Depth ({unit})",
        value=None,
        min_value=0.0,
    )
    survey_bottom_input = well_depth_input
else:
    st.caption(f"Paste MD, Inclination and Azimuth from Excel. MD must be in {unit}.")
    survey_text = st.text_area(
        "Paste survey here",
        value="",
        height=150,
    )

survey_bottom_input = None
if well_profile == "Vertical":
    survey_bottom_input = well_depth_input
else:
    try:
        preview_df = pd.read_csv(StringIO(survey_text), sep=None, engine="python")
        preview_df["MD"] = pd.to_numeric(preview_df["MD"], errors="coerce")
        preview_df = preview_df.dropna(subset=["MD"])
        if not preview_df.empty:
            survey_bottom_input = float(preview_df["MD"].max())
    except Exception:
        survey_bottom_input = None

if survey_bottom_input is not None:
    st.caption(f"Detected survey bottom MD: {survey_bottom_input:,.1f} {unit}")

st.markdown("---")
if is_ct_milling_operation:
    st.markdown("## Milling BHA Data")
    st.caption("Se asume como stuck point la profundidad del mill/bit.")
    fish_depth_label = f"Mill / Top of Fishing MD ({unit})"
    fish_top_input = st.number_input(
        fish_depth_label,
        value=None,
        min_value=0.0,
    )
    stuck_point_input = fish_top_input
else:
    st.markdown("## Fish Data")
    f1, f2 = st.columns(2)
    fish_depth_label = f"Top of Fish - TOF MD ({unit})"
    fish_top_input = f1.number_input(fish_depth_label, value=None, min_value=0.0)
    stuck_point_input = f2.number_input(
        f"Stuck Point MD ({unit}) - optional",
        value=None,
        min_value=0.0,
    )

    st.caption("Enter fish components from top downward.")

default_fish = pd.DataFrame(
    {
        "Component": pd.Series(dtype="object"),
        f"Length ({unit})": pd.Series(dtype="float64"),
        "OD (in)": pd.Series(dtype="float64"),
        "ID (in)": pd.Series(dtype="float64"),
        "Type": pd.Series(dtype="object"),
    }
)

fish_component_options = [
    "Fish",
    "Packer",
    "WRBP",
    "Blanking Plug",
    "Standing Valve",
    "WL Tool",
    "SL Tool",
    "Other",
]

if is_ct_milling_operation:
    fish_df = default_fish.copy()
else:
    fish_df = st.data_editor(
        default_fish,
        num_rows="dynamic",
        use_container_width=True,
        height=185,
        column_config={
            "Component": st.column_config.SelectboxColumn(
                "Component",
                options=fish_component_options,
            ),
        },
        key=f"fish_editor_{unit}_{catalog_application.lower()}_blank_v3",
    )

fish_length_input = None
fish_preview_weight_lbf = None

try:
    if is_ct_milling_operation:
        fish_length_input = 0.0
        fish_preview_weight_lbf = 0.0
    else:
        fish_preview = fish_df.copy()
        fish_length_col = f"Length ({unit})"
        fish_preview[fish_length_col] = pd.to_numeric(
            fish_preview[fish_length_col], errors="coerce"
        ).fillna(0.0)
        fish_preview["OD (in)"] = pd.to_numeric(
            fish_preview["OD (in)"], errors="coerce"
        ).fillna(0.0)
        fish_preview["ID (in)"] = pd.to_numeric(
            fish_preview["ID (in)"], errors="coerce"
        ).fillna(0.0)
        fish_length_input = float(fish_preview[fish_length_col].sum())
        fish_preview["Length_ft"] = fish_preview[fish_length_col].apply(
            lambda x: to_ft(x, unit)
        )
        fish_preview["Weight_lbft"] = fish_preview.apply(
            lambda row: steel_weight_lbft(row["OD (in)"], row["ID (in)"]),
            axis=1,
        )
        fish_preview_weight_lbf = float(
            (fish_preview["Length_ft"] * fish_preview["Weight_lbft"]).sum()
        )
except Exception:
    fish_length_input = None
    fish_preview_weight_lbf = None

if fish_top_input is not None and survey_bottom_input is not None:
    if fish_top_input > survey_bottom_input:
        st.error(
            f"{fish_depth_label} ({fish_top_input:,.1f} {unit}) is deeper than survey bottom "
            f"({survey_bottom_input:,.1f} {unit}). Please check the fish depth or survey."
        )

if (
    fish_top_input is not None
    and fish_length_input is not None
    and survey_bottom_input is not None
):
    fish_bottom_input = fish_top_input + fish_length_input

    if fish_bottom_input > survey_bottom_input:
        st.error(
            f"Fish Bottom MD ({fish_bottom_input:,.1f} {unit}) exceeds survey bottom "
            f"({survey_bottom_input:,.1f} {unit}). "
            f"Fish Top + Fish Length cannot be greater than the maximum MD in the survey."
        )

if not is_ct_milling_operation and fish_length_input is not None and fish_length_input > 0:
    fm1, fm2 = st.columns(2)
    fm1.metric(f"Calculated Fish Length ({unit})", f"{fish_length_input:,.0f}")

    if fish_preview_weight_lbf is not None:
        fm2.metric("Estimated Fish Weight", f"{fish_preview_weight_lbf:,.0f} lbf")

st.markdown("---")
st.markdown("## BHA / Tool String")
if is_ct_job:
    if is_ct_milling_operation:
        st.caption("Enter the CT milling BHA from the mill/bit upward.")
    else:
        st.caption("Enter the CT fishing BHA from the fish upward.")
else:
    st.caption("Enter the fishing BHA from fish upward. Jar and Accelerator rows use the tools selected above.")

tool_catalog = load_tool_catalog()
jar_tool = None
accelerator_tool = None
selected_jar_name = "Manual / Custom"
selected_accelerator_name = "Manual / Custom"

if tool_catalog.empty:
    st.warning("Tool catalog file was not found. Jar and Accelerator rows can still be entered manually.")
else:
    c1, c2 = st.columns(2)

    jar_options = ["Manual / Custom"] + tool_options(
        tool_catalog,
        "Jar",
        catalog_application,
    )
    accelerator_options = ["Manual / Custom", "Not installed"] + tool_options(
        tool_catalog,
        "Accelerator",
        catalog_application,
    )

    selected_jar_name = c1.selectbox(
        "Jar / Martillo",
        jar_options,
        index=0,
    )
    selected_accelerator_name = c2.selectbox(
        "Accelerator / Energizer",
        accelerator_options,
        index=1 if len(accelerator_options) > 1 else 0,
    )

    if selected_jar_name != "Manual / Custom":
        jar_tool = selected_tool(tool_catalog, "Jar", selected_jar_name, catalog_application)

    if selected_accelerator_name not in ["Manual / Custom", "Not installed"]:
        accelerator_tool = selected_tool(
            tool_catalog,
            "Accelerator",
            selected_accelerator_name,
            catalog_application,
        )

    selected_specs = []
    if jar_tool is not None:
        selected_specs.append(
            {
                "Tool": "Jar",
                "Selected": jar_tool["option_label"],
                **tool_capacity_display(jar_tool),
            }
        )
    if accelerator_tool is not None:
        selected_specs.append(
            {
                "Tool": "Accelerator",
                "Selected": accelerator_tool["option_label"],
                **tool_capacity_display(accelerator_tool),
            }
        )

    if selected_specs:
        compact_dataframe(pd.DataFrame(selected_specs), height=120, width_ratio=0.72)

default_bha = pd.DataFrame(
    {
        "Component": pd.Series(dtype="object"),
        "Description": pd.Series(dtype="object"),
        "Joints": pd.Series(dtype="float64"),
        f"Length ({unit})": pd.Series(dtype="float64"),
        "OD (in)": pd.Series(dtype="float64"),
        "ID (in)": pd.Series(dtype="float64"),
        "Type": pd.Series(dtype="object"),
    }
)

default_rows = []
if is_ct_milling_operation:
    default_rows.append({"Component": "Mill/Bit"})
if default_rows:
    default_bha = pd.DataFrame(default_rows).reindex(columns=default_bha.columns)

if selected_accelerator_name == "Not installed":
    default_bha = default_bha[default_bha["Component"] != "Accelerator"].reset_index(drop=True)

default_bha = apply_bha_component_defaults(
    default_bha,
    unit,
    jar_tool,
    accelerator_tool,
)

bha_component_options = (
    (
        [
            "Mill/Bit",
            "CT Connector",
            "Hydraulic Disconnect",
            "Dual Flapper",
            "Circulation Sub",
            "Stabilizer",
            "Weight Bar",
            "Crossover",
            "Jar",
            "Accelerator",
            "Other",
        ]
        if is_ct_milling_operation
        else [
            "Overshot / Spear",
            "CT Connector",
            "Hydraulic Disconnect",
            "Dual Flapper",
            "Circulation Sub",
            "Stabilizer",
            "Weight Bar",
            "Crossover",
            "Jar",
            "Accelerator",
            "Other",
        ]
    )
    if is_ct_job
    else [
        "Overshot / Spear",
        "Safety Joint",
        "Bumper Sub",
        "Drill Collar",
        "HWDP",
        "Drill Pipe",
        "Crossover",
        "Jar",
        "Accelerator",
        "Other",
    ]
)

bha_mode_key = "milling" if is_ct_milling_operation else "fishing"
bha_state_key = f"bha_data_{unit}_{catalog_application.lower()}_{bha_mode_key}_blank_v9"
bha_version_key = f"bha_editor_version_{unit}_{catalog_application.lower()}_{bha_mode_key}_blank_v9"

if bha_state_key not in st.session_state:
    st.session_state[bha_state_key] = default_bha

if bha_version_key not in st.session_state:
    st.session_state[bha_version_key] = 0

prepared_bha = apply_bha_component_defaults(
    st.session_state[bha_state_key],
    unit,
    jar_tool,
    accelerator_tool,
)

if not dataframes_match(prepared_bha, st.session_state[bha_state_key]):
    st.session_state[bha_state_key] = prepared_bha
    st.session_state[bha_version_key] += 1

editor_key = (
    f"bha_editor_{unit}_{catalog_application.lower()}_"
    f"{bha_mode_key}_{st.session_state[bha_version_key]}"
)


def sync_bha_editor_defaults():
    editor_state = st.session_state.get(editor_key)
    if editor_state is None:
        return

    if isinstance(editor_state, pd.DataFrame):
        edited_work = editor_state.copy()
    else:
        edited_work = st.session_state[bha_state_key].copy()
        for row_index in editor_state.get("deleted_rows", []):
            if row_index in edited_work.index:
                edited_work = edited_work.drop(index=row_index)

        for row_index, changes in editor_state.get("edited_rows", {}).items():
            row_index = int(row_index)
            for column, value in changes.items():
                if column not in edited_work.columns:
                    edited_work[column] = None
                edited_work.loc[row_index, column] = value

        added_rows = editor_state.get("added_rows", [])
        if added_rows:
            edited_work = pd.concat(
                [edited_work, pd.DataFrame(added_rows)],
                ignore_index=True,
                sort=False,
            )
        else:
            edited_work = edited_work.reset_index(drop=True)

    edited_work = apply_bha_component_defaults(
        edited_work,
        unit,
        jar_tool,
        accelerator_tool,
    )
    st.session_state[bha_state_key] = edited_work
    st.session_state[bha_version_key] += 1


bha_df = st.data_editor(
    st.session_state[bha_state_key],
    num_rows="dynamic",
    use_container_width=True,
    height=280,
    hide_index=True,
    column_order=[
        "Component",
        "Description",
        "Joints",
        f"Length ({unit})",
        "OD (in)",
        "ID (in)",
    ],
    column_config={
        "Component": st.column_config.SelectboxColumn(
            "Component",
            options=bha_component_options,
            required=True,
        ),
        "Description": st.column_config.TextColumn(
            "Description",
            help="Auto-filled for standard components. Use this field to name Other items.",
        ),
        "Joints": st.column_config.NumberColumn(
            "Joints",
            help="Used for Drill Collar, HWDP and Weight Bar rows. Length is calculated as joints x 30 ft.",
            min_value=0,
            step=1,
        ),
    },
    key=editor_key,
    on_change=sync_bha_editor_defaults,
)

filled_bha_df = apply_bha_component_defaults(
    bha_df,
    unit,
    jar_tool,
    accelerator_tool,
)

if (
    not dataframes_match(filled_bha_df, bha_df)
    or not dataframes_match(filled_bha_df, st.session_state[bha_state_key])
):
    st.session_state[bha_state_key] = filled_bha_df
    st.session_state[bha_version_key] += 1
    st.rerun()

bha_df = filled_bha_df

if selected_accelerator_name == "Not installed":
    accelerator_rows = bha_df["Component"].astype(str).str.lower() == "accelerator"
    if accelerator_rows.any():
        st.session_state[bha_state_key] = bha_df[~accelerator_rows].reset_index(drop=True)
        st.session_state[bha_version_key] += 1
        st.rerun()

st.markdown("---")
st.markdown("## Work String")
st.caption("Work string from the top of the BHA to surface.")

ws1, ws2, ws3 = st.columns(3)
work_string_options = (
    ["Select...", "Coiled Tubing"]
    if is_ct_job
    else ["Select...", "Drill Pipe", "Tubing"]
)
work_string_type = ws1.selectbox(
    "Work String Type",
    work_string_options,
)

work_string_od = None
work_string_id = None
work_string_weight_lbft = None
work_string_description = ""

if work_string_type != "Select...":
    work_catalog = work_string_catalog(work_string_type)
    work_string_od_label = ws2.selectbox(
        "OD",
        ["Select..."] + work_catalog["OD_label"].drop_duplicates().tolist(),
    )

    if work_string_od_label != "Select...":
        available_weights = work_catalog[work_catalog["OD_label"] == work_string_od_label][
            "WT_label"
        ].dropna().tolist()
        selected_work_string_label = ws3.selectbox(
            "Wall Thickness (in)" if work_string_type == "Coiled Tubing" else "Nominal Weight (lb/ft)",
            ["Select..."] + available_weights,
        )

        if selected_work_string_label != "Select...":
            selected_work_string = work_catalog[
                (work_catalog["OD_label"] == work_string_od_label)
                & (work_catalog["WT_label"] == selected_work_string_label)
            ].iloc[0]
            selected_work_string_wt = float(selected_work_string["WT"])
            work_string_od = float(selected_work_string["OD_numeric"])
            work_string_id = float(selected_work_string["ID"])
            work_string_weight_lbft = float(selected_work_string["Weight_lbft"])
            if work_string_type == "Coiled Tubing":
                work_string_description = (
                    f"{work_string_type} {work_string_od_label} OD x "
                    f"{selected_work_string_wt:g} in wall"
                )
            else:
                work_string_description = (
                    f"{work_string_type} {work_string_od_label} OD, "
                    f"{work_string_weight_lbft:g} lb/ft"
                )

            ws_m1, ws_m2, ws_m3 = st.columns(3)
            ws_m1.metric("Selected Work String", work_string_description)
            ws_m2.metric("Work String ID", f'{work_string_id:g}"')
            ws_m3.metric(
                "Nominal Weight",
                format_report_number(work_string_weight_lbft, 3) + " lb/ft",
            )

st.markdown("---")
st.markdown("## Wellbore Data")
st.caption("Used with the survey to estimate contact/friction energy losses.")

wb1, wb2, wb3 = st.columns(3)
wellbore_options = (
    ["Select...", "Casing", "Open Hole", "Drill Pipe", "Tubing"]
    if is_ct_job
    else ["Select...", "Cased Hole", "Open Hole"]
)
wellbore_type = wb1.selectbox(
    "Hole Type",
    wellbore_options,
)

hole_id_in = None
wellbore_description = ""

if wellbore_type in ["Cased Hole", "Casing"]:
    casing_data = casing_catalog()
    casing_od_label = wb2.selectbox(
        "Casing OD",
        ["Select..."] + casing_data["OD_label"].drop_duplicates().tolist(),
    )

    if casing_od_label != "Select...":
        casing_weights = casing_data[casing_data["OD_label"] == casing_od_label][
            "WT"
        ].dropna().tolist()
        casing_weight = wb3.selectbox(
            "Casing Weight (lb/ft)",
            ["Select..."] + casing_weights,
            format_func=lambda x: x if isinstance(x, str) else f"{x:g}",
        )
        if casing_weight != "Select...":
            selected_casing = casing_data[
                (casing_data["OD_label"] == casing_od_label)
                & (casing_data["WT"] == casing_weight)
            ].iloc[0]
            hole_id_in = float(selected_casing["ID"])
            wellbore_description = (
                f"Cased Hole - {casing_od_label} {casing_weight:g} lb/ft casing"
            )
            st.metric("Selected Casing ID", f'{hole_id_in:.2f}"')
elif wellbore_type == "Open Hole":
    hole_id_in = wb2.number_input(
        "Open Hole ID (in)",
        value=None,
        min_value=0.0,
    )
    wellbore_description = "Open Hole"
elif wellbore_type in ["Drill Pipe", "Tubing"]:
    tubular_data = work_string_catalog(wellbore_type)
    tubular_od_label = wb2.selectbox(
        f"{wellbore_type} OD",
        ["Select..."] + tubular_data["OD_label"].drop_duplicates().tolist(),
    )
    if tubular_od_label != "Select...":
        tubular_weights = tubular_data[tubular_data["OD_label"] == tubular_od_label][
            "WT_label"
        ].dropna().tolist()
        tubular_weight_label = wb3.selectbox(
            f"{wellbore_type} Weight (lb/ft)",
            ["Select..."] + tubular_weights,
        )
        if tubular_weight_label != "Select...":
            selected_tubular = tubular_data[
                (tubular_data["OD_label"] == tubular_od_label)
                & (tubular_data["WT_label"] == tubular_weight_label)
            ].iloc[0]
            tubular_weight = float(selected_tubular["WT"])
            hole_id_in = float(selected_tubular["ID"])
            wellbore_description = (
                f"{wellbore_type} - {tubular_od_label} {tubular_weight:g} lb/ft"
            )
            st.metric(f"Selected {wellbore_type} ID", f'{hole_id_in:.2f}"')

st.markdown("---")
st.markdown("## Impact Simulation Inputs")
st.caption("Physical inputs used to estimate up-jarring impact and generate the report.")

is1, is2 = st.columns(2)
mud_weight_ppg = is1.number_input("Mud Weight (lb/gal)", value=None, min_value=0.0)
friction_coefficient = None

if wellbore_type == "Select...":
    is2.info("Select Hole Type to enable friction range.")
else:
    is_open_hole = wellbore_type == "Open Hole"
    friction_min = 0.25 if is_open_hole else 0.0
    friction_max = 0.40 if is_open_hole else 0.25
    friction_default = 0.30 if is_open_hole else 0.20

    if "friction_coefficient_blank_v1" not in st.session_state:
        st.session_state["friction_coefficient_blank_v1"] = friction_default
    if st.session_state["friction_coefficient_blank_v1"] < friction_min:
        st.session_state["friction_coefficient_blank_v1"] = friction_min
    if st.session_state["friction_coefficient_blank_v1"] > friction_max:
        st.session_state["friction_coefficient_blank_v1"] = friction_max

    friction_coefficient = is2.number_input(
        f"Friction Coefficient ({friction_min:.2f} - {friction_max:.2f})",
        min_value=friction_min,
        max_value=friction_max,
        step=0.01,
        key="friction_coefficient_blank_v1",
    )

jar_pull_limit_lbf = jar_activation_limit_lbf(jar_tool, "up")
jar_push_limit_lbf = jar_activation_limit_lbf(jar_tool, "down") if is_ct_job else None
applied_overpull_default_lbf = None
if jar_pull_limit_lbf is not None:
    st.caption(
        f"Selected jar pull limit for analysis: {jar_pull_limit_lbf:,.0f} lb. "
        "Applied overpull cannot exceed this value."
    )
    applied_overpull_default_lbf = None

if (
    jar_pull_limit_lbf is not None
    and st.session_state.get("applied_overpull_lbf_blank_v1") is not None
    and st.session_state["applied_overpull_lbf_blank_v1"] > jar_pull_limit_lbf
):
    st.session_state["applied_overpull_lbf_blank_v1"] = jar_pull_limit_lbf

applied_overpull_lbf = st.number_input(
    "Applied Overpull / Firing Load Up (lb)",
    value=applied_overpull_default_lbf,
    min_value=0.0,
    max_value=jar_pull_limit_lbf,
    step=5000.0,
    key="applied_overpull_lbf_blank_v1",
)

applied_setdown_lbf = None
if is_ct_job:
    if jar_push_limit_lbf is not None:
        st.caption(
            f"Selected CT jar push limit for down-jarring analysis: {jar_push_limit_lbf:,.0f} lb. "
            "Applied setdown cannot exceed this value."
        )
    if (
        jar_push_limit_lbf is not None
        and st.session_state.get("applied_setdown_lbf_blank_v1") is not None
        and st.session_state["applied_setdown_lbf_blank_v1"] > jar_push_limit_lbf
    ):
        st.session_state["applied_setdown_lbf_blank_v1"] = jar_push_limit_lbf

    applied_setdown_lbf = st.number_input(
        "Applied Setdown / Firing Load Down (lb)",
        value=None,
        min_value=0.0,
        max_value=jar_push_limit_lbf,
        step=1000.0,
        key="applied_setdown_lbf_blank_v1",
    )

if st.button("Generate Well Path and BHA Placement"):
    st.session_state["run_simulation"] = True

if st.session_state.get("run_simulation"):
    try:
        ct_mill_mode = is_ct_milling_operation

        if work_string_od is None or work_string_id is None or work_string_weight_lbft is None:
            st.error("Work String selection is required.")
            st.stop()

        if wellbore_type == "Select..." or hole_id_in is None:
            st.error("Wellbore Data is required.")
            st.stop()

        if mud_weight_ppg is None:
            st.error("Mud Weight is required.")
            st.stop()

        if friction_coefficient is None:
            st.error("Friction Coefficient is required.")
            st.stop()

        if applied_overpull_lbf is None:
            st.error("Applied Overpull / Firing Load Up is required.")
            st.stop()

        if is_ct_job and applied_setdown_lbf is None:
            st.error("Applied Setdown / Firing Load Down is required for CT jar simulations.")
            st.stop()

        if jar_pull_limit_lbf is not None and applied_overpull_lbf > jar_pull_limit_lbf:
            st.error(
                f"Applied overpull cannot exceed the selected jar limit of "
                f"{jar_pull_limit_lbf:,.0f} lb."
            )
            st.stop()

        if is_ct_job and jar_push_limit_lbf is not None and applied_setdown_lbf > jar_push_limit_lbf:
            st.error(
                f"Applied setdown cannot exceed the selected CT jar push limit of "
                f"{jar_push_limit_lbf:,.0f} lb."
            )
            st.stop()

        if well_profile == "Vertical":
            well_depth_ft = to_ft(well_depth_input, unit)
            if well_depth_ft is None or well_depth_ft <= 0:
                st.error("Well Depth is required for a vertical well.")
                st.stop()
            survey_df = pd.DataFrame(
                {
                    "MD": [0.0, well_depth_ft],
                    "Inclination": [0.0, 0.0],
                    "Azimuth": [0.0, 0.0],
                }
            )
        else:
            survey_df = pd.read_csv(StringIO(survey_text), sep=None, engine="python")

            if unit == "m":
                survey_df["MD"] = pd.to_numeric(survey_df["MD"], errors="coerce") * M_TO_FT

        trajectory = calculate_trajectory_minimum_curvature(survey_df)

        fish_top_ft = to_ft(fish_top_input, unit)
        stuck_point_ft = (
            to_ft(stuck_point_input, unit)
            if stuck_point_input is not None
            else fish_top_ft
        )

        if fish_top_ft is None:
            st.error(f"{fish_depth_label} is required.")
            st.stop()

        if fish_top_ft > trajectory["MD"].max():
            st.error(
                f"Fish Top MD is deeper than survey bottom. "
                f"Fish Top: {from_ft(fish_top_ft, unit):,.1f} {unit}; "
                f"Survey Bottom: {from_ft(trajectory['MD'].max(), unit):,.1f} {unit}."
            )
            st.stop()

        length_col = f"Length ({unit})"
        if ct_mill_mode:
            fish_work = pd.DataFrame(
                [
                    {
                        "Component": "Mill/Bit impact point",
                        length_col: 0.0,
                        "OD (in)": 0.0,
                        "ID (in)": 0.0,
                        "Type": "CT Milling",
                        "Length_ft": 0.0,
                    }
                ]
            )
            fish_length_ft = 0.0
        else:
            fish_work = fish_df.copy()
            fish_work[length_col] = pd.to_numeric(
                fish_work[length_col], errors="coerce"
            ).fillna(0.0)
            fish_work["OD (in)"] = pd.to_numeric(
                fish_work["OD (in)"], errors="coerce"
            ).fillna(0.0)
            fish_work["ID (in)"] = pd.to_numeric(
                fish_work["ID (in)"], errors="coerce"
            ).fillna(0.0)

            fish_work = fish_work[fish_work[length_col] > 0].copy()

            if fish_work.empty:
                st.error("At least one fish component with a positive length is required.")
                st.stop()

            fish_work["Length_ft"] = fish_work[length_col].apply(lambda x: to_ft(x, unit))
            fish_length_ft = float(fish_work["Length_ft"].sum())

        fish_bottom_ft = fish_top_ft + fish_length_ft

        if fish_bottom_ft > trajectory["MD"].max():
            st.error(
                f"Fish Bottom MD exceeds the survey bottom. "
                f"Fish Top + Fish Length = {from_ft(fish_bottom_ft, unit):,.1f} {unit}; "
                f"Survey Bottom = {from_ft(trajectory['MD'].max(), unit):,.1f} {unit}. "
                f"Please reduce the fish length, adjust the fish top, or extend the survey."
            )
            st.stop()

        fish_top_mds = []
        fish_bottom_mds = []
        fish_center_mds = []

        current_top = fish_top_ft

        for _, row in fish_work.iterrows():
            comp_length = row["Length_ft"]
            comp_top = current_top
            comp_bottom = comp_top + comp_length
            comp_center = (comp_top + comp_bottom) / 2.0

            fish_top_mds.append(comp_top)
            fish_bottom_mds.append(comp_bottom)
            fish_center_mds.append(comp_center)

            current_top = comp_bottom

        fish_work["Top_MD_ft"] = fish_top_mds
        fish_work["Bottom_MD_ft"] = fish_bottom_mds
        fish_work["Center_MD_ft"] = fish_center_mds
        fish_work["Weight_lbft"] = fish_work.apply(
            lambda row: steel_weight_lbft(row["OD (in)"], row["ID (in)"]),
            axis=1,
        )
        fish_work["Component Weight (lbf)"] = (
            fish_work["Length_ft"] * fish_work["Weight_lbft"]
        )
        fish_total_weight_lbf = float(fish_work["Component Weight (lbf)"].sum())

        if unit == "m":
            fish_work["Top MD (m)"] = fish_work["Top_MD_ft"] * FT_TO_M
            fish_work["Bottom MD (m)"] = fish_work["Bottom_MD_ft"] * FT_TO_M
            fish_work["Center MD (m)"] = fish_work["Center_MD_ft"] * FT_TO_M
        else:
            fish_work["Top MD (ft)"] = fish_work["Top_MD_ft"]
            fish_work["Bottom MD (ft)"] = fish_work["Bottom_MD_ft"]
            fish_work["Center MD (ft)"] = fish_work["Center_MD_ft"]

        bha_work = bha_df.copy()
        bha_length_col = f"Length ({unit})"

        bha_work[bha_length_col] = pd.to_numeric(
            bha_work[bha_length_col], errors="coerce"
        ).fillna(0.0)
        bha_work["OD (in)"] = pd.to_numeric(
            bha_work["OD (in)"], errors="coerce"
        ).fillna(0.0)
        bha_work["ID (in)"] = pd.to_numeric(
            bha_work["ID (in)"], errors="coerce"
        ).fillna(0.0)
        bha_work["Joints"] = pd.to_numeric(
            bha_work.get("Joints", 0.0), errors="coerce"
        ).fillna(0.0)

        bha_work["Length_ft"] = bha_work[bha_length_col].apply(lambda x: to_ft(x, unit))
        bha_work["Weight_lbft"] = bha_work.apply(
            lambda row: steel_weight_lbft(row["OD (in)"], row["ID (in)"]),
            axis=1,
        )

        top_mds = []
        bottom_mds = []
        center_mds = []

        current_bottom = fish_top_ft

        for _, row in bha_work.iterrows():
            comp_length = row["Length_ft"]
            comp_bottom = current_bottom
            comp_top = comp_bottom - comp_length
            comp_center = (comp_top + comp_bottom) / 2.0

            top_mds.append(comp_top)
            bottom_mds.append(comp_bottom)
            center_mds.append(comp_center)

            current_bottom = comp_top

        bha_top_ft = current_bottom
        work_string_length_ft = max(bha_top_ft, 0.0)
        work_string_total_weight_lbf = (
            work_string_length_ft * work_string_weight_lbft
        )

        if bha_top_ft < 0:
            st.warning(
                "The BHA length extends above surface. Work string length was set to 0. "
                "Please review the BHA component lengths."
            )

        bha_work["Top_MD_ft"] = top_mds
        bha_work["Bottom_MD_ft"] = bottom_mds
        bha_work["Center_MD_ft"] = center_mds
        bha_work["Component Weight (lbf)"] = (
            bha_work["Length_ft"] * bha_work["Weight_lbft"]
        )

        if unit == "m":
            bha_work["Top MD (m)"] = bha_work["Top_MD_ft"] * FT_TO_M
            bha_work["Bottom MD (m)"] = bha_work["Bottom_MD_ft"] * FT_TO_M
            bha_work["Center MD (m)"] = bha_work["Center_MD_ft"] * FT_TO_M
        else:
            bha_work["Top MD (ft)"] = bha_work["Top_MD_ft"]
            bha_work["Bottom MD (ft)"] = bha_work["Bottom_MD_ft"]
            bha_work["Center MD (ft)"] = bha_work["Center_MD_ft"]

        jar_rows = bha_work[bha_work["Type"].astype(str).str.lower().str.contains("jar")]
        acc_rows = bha_work[
            bha_work["Type"].astype(str).str.lower().str.contains("accelerator")
        ]

        if jar_tool is not None and len(jar_rows) == 0:
            st.error(
                "You selected a jar above, but no BHA row has Component = Jar. "
                "Add a Jar row where you want that tool placed in the BHA."
            )
            st.stop()

        if accelerator_tool is not None and len(acc_rows) == 0:
            st.error(
                "You selected an accelerator above, but no BHA row has Component = Accelerator. "
                "Add an Accelerator row where you want that tool placed in the BHA."
            )
            st.stop()

        dc_rows = bha_work[bha_work["Type"].astype(str).str.lower().eq("dc")]
        weight_bar_rows = bha_work[
            bha_work["Type"].astype(str).str.lower().eq("weight bar")
        ]
        if len(jar_rows) > 0:
            first_jar_index = jar_rows.index[0]
            dc_rows_above_jar = dc_rows[dc_rows.index > first_jar_index]
            weight_bar_rows_above_jar = weight_bar_rows[
                weight_bar_rows.index > first_jar_index
            ]
            bha_rows_below_jar = bha_work[bha_work.index < first_jar_index]
        else:
            dc_rows_above_jar = dc_rows
            weight_bar_rows_above_jar = weight_bar_rows
            bha_rows_below_jar = bha_work.iloc[0:0]

        jar_md_ft = float(jar_rows["Center_MD_ft"].iloc[0]) if len(jar_rows) > 0 else None
        acc_md_ft = float(acc_rows["Center_MD_ft"].iloc[0]) if len(acc_rows) > 0 else None
        jar_stroke_in = (
            float(jar_tool["stroke_in"])
            if jar_tool is not None and not pd.isna(jar_tool.get("stroke_in"))
            else 0.0
        )
        accelerator_stroke_in = (
            float(accelerator_tool["stroke_in"])
            if accelerator_tool is not None and not pd.isna(accelerator_tool.get("stroke_in"))
            else 0.0
        )
        accelerator_weight_lbf = (
            float(acc_rows["Component Weight (lbf)"].sum()) if len(acc_rows) > 0 else 0.0
        )
        if is_ct_job:
            if len(jar_rows) > 0 and len(acc_rows) > 0:
                first_acc_index = acc_rows.index[0]
                lower_index = min(first_jar_index, first_acc_index)
                upper_index = max(first_jar_index, first_acc_index)
                sensitivity_rows_above_jar = weight_bar_rows[
                    (weight_bar_rows.index > lower_index)
                    & (weight_bar_rows.index < upper_index)
                ]
            else:
                sensitivity_rows_above_jar = weight_bar_rows_above_jar
            max_joints_above_jar_work = int(
                max(float(sensitivity_rows_above_jar["Joints"].sum()), 0.0)
            )
        else:
            sensitivity_rows_above_jar = dc_rows_above_jar
            max_joints_above_jar_work = int(
                max(float(sensitivity_rows_above_jar["Joints"].sum()), 1.0)
            )
        has_joint_sensitivity = max_joints_above_jar_work > 0
        joint_length_ft = 30.0
        joint_weight_lbft = (
            float(
                sensitivity_rows_above_jar["Weight_lbft"]
                .replace(0, np.nan)
                .dropna()
                .mean()
            )
            if len(sensitivity_rows_above_jar) > 0
            and not sensitivity_rows_above_jar["Weight_lbft"]
            .replace(0, np.nan)
            .dropna()
            .empty
            else work_string_weight_lbft
        )
        steel_modulus_psi = 30000000.0
        stopping_distance_in = max(jar_stroke_in * 0.25, 2.0 if is_ct_job else 2.5)
        applied_firing_load_lbf = applied_overpull_lbf

        if ct_mill_mode:
            fish_length_ft = float(bha_rows_below_jar["Length_ft"].sum())
            fish_total_weight_lbf = float(
                bha_rows_below_jar["Component Weight (lbf)"].sum()
            )

        trajectory_efficiency, trajectory_loss_details = estimate_survey_efficiency(
            trajectory,
            friction_coefficient,
        )
        clearance_efficiency, clearance_details = estimate_clearance_efficiency(
            hole_id_in,
            work_string_od,
            bha_work,
        )
        fluid_efficiency = max(0.85, min(1.05, 1.0 - (mud_weight_ppg - 8.4) * 0.01))

        ct_setdown_available_lbf = None
        ct_buckling_details = {}
        applied_firing_load_down_lbf = applied_setdown_lbf
        if is_ct_job:
            local_weight_above_jar_lbf = float(
                sensitivity_rows_above_jar["Component Weight (lbf)"].sum()
            )
            ct_setdown_available_lbf, ct_buckling_details = estimate_ct_setdown_capacity(
                trajectory,
                hole_id_in,
                work_string_od,
                work_string_id,
                work_string_weight_lbft,
                mud_weight_ppg,
                friction_coefficient,
                steel_modulus_psi,
                local_weight_above_jar_lbf,
            )
            setdown_limit_candidates = [
                value
                for value in [ct_setdown_available_lbf, jar_push_limit_lbf]
                if value is not None and not pd.isna(value) and value > 0
            ]
            effective_setdown_limit_lbf = (
                min(setdown_limit_candidates) if setdown_limit_candidates else 0.0
            )
            applied_firing_load_down_lbf = min(
                applied_setdown_lbf,
                effective_setdown_limit_lbf,
            )
            if applied_setdown_lbf > applied_firing_load_down_lbf:
                st.warning(
                    f"Requested down firing load was {applied_setdown_lbf:,.0f} lb, "
                    f"but CT buckling / jar limits reduce the available setdown at the jar "
                    f"to {applied_firing_load_down_lbf:,.0f} lb."
                )

        if clearance_details.get("Minimum radial clearance (in)") is not None:
            if clearance_details["Minimum radial clearance (in)"] <= 0:
                st.warning(
                    "The selected Hole / Casing ID is smaller than, or equal to, the maximum OD in the string."
                )

        physics_inputs = {
            "work_string_length_ft": work_string_length_ft,
            "work_string_od_in": work_string_od,
            "work_string_id_in": work_string_id,
            "work_string_weight_lbft": work_string_weight_lbft,
            "joint_length_ft": joint_length_ft,
            "joint_weight_lbft": joint_weight_lbft,
            "fish_weight_lbf": fish_total_weight_lbf,
            "fish_length_ft": fish_length_ft,
            "jar_stroke_in": jar_stroke_in,
            "accelerator_stroke_in": accelerator_stroke_in,
            "accelerator_weight_lbf": accelerator_weight_lbf,
            "trajectory_efficiency": trajectory_efficiency,
            "clearance_efficiency": clearance_efficiency,
            "fluid_efficiency": fluid_efficiency,
            "steel_modulus_psi": steel_modulus_psi,
            "stopping_distance_in": stopping_distance_in,
            "effective_area_floor_in2": 2.0 if is_ct_job else 4.0,
            "accelerator_energy_factor": 0.75 if is_ct_job else 0.5,
        }

        impact_by_joints = calculate_impact_by_joints(
            applied_firing_load_lbf,
            max_joints_above_jar_work,
            physics_inputs,
        )
        impact_by_joints_down = None
        if is_ct_job:
            impact_by_joints_down = calculate_impact_by_joints(
                applied_firing_load_down_lbf,
                max_joints_above_jar_work,
                physics_inputs,
            ).rename(
                columns={
                    "Impact (lb)": "Down Impact (lb)",
                    "Impulse (lb-sec)": "Down Impulse (lb-sec)",
                    "Up Impact Ratio": "Down Impact Ratio",
                }
            )

        fish_top_point = interpolate_at_md(trajectory, fish_top_ft)

        st.markdown("## Fish Summary")

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(f"Fish Top MD ({unit})", f"{from_ft(fish_top_ft, unit):,.0f}")
        s2.metric(f"Fish Bottom MD ({unit})", f"{from_ft(fish_bottom_ft, unit):,.0f}")
        s3.metric("Fish Weight", f"{fish_total_weight_lbf:,.0f} lbf")
        s4.metric(f"Stuck Point MD ({unit})", f"{from_ft(stuck_point_ft, unit):,.0f}")

        st.markdown("## Calculated Fish Components")

        fish_display_cols = [
            "Component",
            "Type",
            length_col,
            "OD (in)",
            "ID (in)",
            "Weight_lbft",
            "Component Weight (lbf)",
        ]

        if unit == "m":
            fish_display_cols += ["Top MD (m)", "Bottom MD (m)", "Center MD (m)"]
        else:
            fish_display_cols += ["Top MD (ft)", "Bottom MD (ft)", "Center MD (ft)"]

        st.dataframe(
            format_display_dataframe(
                fish_work[fish_display_cols],
                {
                    "OD (in)": 3,
                    "ID (in)": 3,
                    "Weight_lbft": 3,
                },
            ),
            use_container_width=True,
            height=220,
        )

        st.markdown("## Calculated BHA Placement")

        display_cols = [
            "Component",
            "Description",
            "Joints",
            bha_length_col,
            "OD (in)",
            "ID (in)",
            "Weight_lbft",
            "Component Weight (lbf)",
        ]

        if unit == "m":
            display_cols += ["Top MD (m)", "Bottom MD (m)", "Center MD (m)"]
        else:
            display_cols += ["Top MD (ft)", "Bottom MD (ft)", "Center MD (ft)"]

        st.dataframe(
            format_display_dataframe(
                bha_work[display_cols],
                {
                    "OD (in)": 3,
                    "ID (in)": 3,
                    "Weight_lbft": 3,
                },
            ),
            use_container_width=True,
            height=260,
        )

        st.markdown("## Work String Summary")

        ws_summary_1, ws_summary_2, ws_summary_3, ws_summary_4 = st.columns(4)
        ws_summary_1.metric(
            f"Work String Length ({unit})",
            f"{from_ft(work_string_length_ft, unit):,.0f}",
        )
        ws_summary_2.metric("Nominal Weight", format_report_number(work_string_weight_lbft, 3) + " lb/ft")
        ws_summary_3.metric(
            "Work String Weight",
            f"{work_string_total_weight_lbf:,.0f} lbf",
        )
        ws_summary_4.metric(
            f"BHA Top MD ({unit})",
            f"{from_ft(bha_top_ft, unit):,.0f}",
        )

        work_string_display = pd.DataFrame(
            [
                {
                    "Description": work_string_description,
                    f"Length ({unit})": from_ft(work_string_length_ft, unit),
                    "OD (in)": work_string_od,
                    "ID (in)": work_string_id,
                    "Weight_lbft": work_string_weight_lbft,
                    "Total Weight (lbf)": work_string_total_weight_lbf,
                    f"Bottom MD ({unit})": from_ft(work_string_length_ft, unit),
                    f"Top MD ({unit})": 0.0,
                }
            ]
        )
        compact_dataframe(
            work_string_display,
            height=100,
            decimal_cols={
                "OD (in)": 3,
                "ID (in)": 3,
                "Weight_lbft": 3,
            },
            width_ratio=0.70,
        )

        p1, p2 = st.columns(2)

        if jar_md_ft:
            p1.metric(f"Calculated Jar MD ({unit})", f"{from_ft(jar_md_ft, unit):,.0f}")
        else:
            p1.metric("Calculated Jar MD", "Not found")

        if acc_md_ft:
            p2.metric(
                f"Calculated Accelerator MD ({unit})",
                f"{from_ft(acc_md_ft, unit):,.0f}",
            )
        else:
            p2.metric("Calculated Accelerator MD", "Not installed")

        st.markdown("## Calculated Trajectory")
        trajectory_display = trajectory.copy()

        if unit == "m":
            trajectory_display["MD"] = trajectory_display["MD"] * FT_TO_M
            trajectory_display["TVD"] = trajectory_display["TVD"] * FT_TO_M
            trajectory_display["Northing"] = trajectory_display["Northing"] * FT_TO_M
            trajectory_display["Easting"] = trajectory_display["Easting"] * FT_TO_M

        st.dataframe(
            format_display_dataframe(
                trajectory_display,
                {
                    "Inclination": 3,
                    "Azimuth": 3,
                    "DLS": 3,
                },
            ),
            use_container_width=True,
            height=260,
        )

        x_east = trajectory["Easting"] * FT_TO_M if unit == "m" else trajectory["Easting"]
        y_north = trajectory["Northing"] * FT_TO_M if unit == "m" else trajectory["Northing"]
        z_tvd = trajectory["TVD"] * FT_TO_M if unit == "m" else trajectory["TVD"]
        md_plot = trajectory["MD"] * FT_TO_M if unit == "m" else trajectory["MD"]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=x_east,
                y=y_north,
                z=-z_tvd,
                mode="lines",
                line=dict(width=6, color="#008FE3"),
                name="Well Path",
            )
        )

        add_marker_3d(fig, fish_top_point, "Fish Top", "red", unit)

        fig.update_layout(
            title="3D Well Trajectory",
            scene=dict(
                xaxis_title=f"Easting ({unit})",
                yaxis_title=f"Northing ({unit})",
                zaxis_title=f"TVD ({unit})",
                aspectmode="cube",
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=32),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## 2D Well Path Views")

        v1, v2 = st.columns(2)

        fig_top = go.Figure()
        fig_top.add_trace(
            go.Scatter(
                x=x_east,
                y=y_north,
                mode="lines",
                line=dict(width=3, color="#008FE3"),
                name="Well Path",
            )
        )

        add_marker_2d(
            fig_top,
            fish_top_point,
            "Easting",
            "Northing",
            "Fish Top",
            "red",
            unit,
            negative_y=False,
        )

        fig_top.update_layout(
            title="Top View / Aerial View",
            xaxis_title=f"Easting ({unit})",
            yaxis_title=f"Northing ({unit})",
            height=360,
            margin=dict(l=10, r=10, b=32, t=42),
            xaxis=dict(scaleanchor="y", scaleratio=1),
        )

        v1.plotly_chart(fig_top, use_container_width=True)

        fig_side_east = go.Figure()
        fig_side_east.add_trace(
            go.Scatter(
                x=x_east,
                y=-z_tvd,
                mode="lines",
                line=dict(width=3, color="#F28C00"),
                name="Easting vs TVD",
            )
        )

        add_marker_2d(
            fig_side_east,
            fish_top_point,
            "Easting",
            "TVD",
            "Fish Top",
            "red",
            unit,
        )

        fig_side_east.update_layout(
            title="Side View - Easting vs TVD",
            xaxis_title=f"Easting ({unit})",
            yaxis_title=f"TVD ({unit})",
            height=360,
            margin=dict(l=10, r=10, b=32, t=42),
        )

        v2.plotly_chart(fig_side_east, use_container_width=True)

        v3, v4 = st.columns(2)

        fig_side_north = go.Figure()
        fig_side_north.add_trace(
            go.Scatter(
                x=y_north,
                y=-z_tvd,
                mode="lines",
                line=dict(width=3, color="#6A3D9A"),
                name="Northing vs TVD",
            )
        )

        add_marker_2d(
            fig_side_north,
            fish_top_point,
            "Northing",
            "TVD",
            "Fish Top",
            "red",
            unit,
        )

        fig_side_north.update_layout(
            title="Side View - Northing vs TVD",
            xaxis_title=f"Northing ({unit})",
            yaxis_title=f"TVD ({unit})",
            height=360,
            margin=dict(l=10, r=10, b=32, t=42),
        )

        v3.plotly_chart(fig_side_north, use_container_width=True)

        fig_md_tvd = go.Figure()
        fig_md_tvd.add_trace(
            go.Scatter(
                x=md_plot,
                y=-z_tvd,
                mode="lines",
                line=dict(width=3, color="#D62728"),
                name="MD vs TVD",
            )
        )

        add_marker_2d(
            fig_md_tvd,
            fish_top_point,
            "MD",
            "TVD",
            "Fish Top",
            "red",
            unit,
        )

        fig_md_tvd.update_layout(
            title="Vertical Section - MD vs TVD",
            xaxis_title=f"Measured Depth ({unit})",
            yaxis_title=f"TVD ({unit})",
            height=360,
            margin=dict(l=10, r=10, b=32, t=42),
        )

        v4.plotly_chart(fig_md_tvd, use_container_width=True)

        st.markdown("## True Vertical Section")

        horizontal_departure = np.sqrt(
            trajectory["Easting"] ** 2 + trajectory["Northing"] ** 2
        )

        if unit == "m":
            horizontal_departure = horizontal_departure * FT_TO_M

        tvd_max_plot = float(np.nanmax(z_tvd)) if len(z_tvd) else 0.0
        departure_min_plot = float(np.nanmin(horizontal_departure)) if len(horizontal_departure) else 0.0
        departure_max_plot = float(np.nanmax(horizontal_departure)) if len(horizontal_departure) else 0.0
        departure_span_plot = max(departure_max_plot - departure_min_plot, 0.0)
        horizontal_padding = max(
            tvd_max_plot * 0.08,
            departure_span_plot * 0.15,
            100.0 if unit == "ft" else 30.0,
        )
        departure_x_range = [
            min(0.0, departure_min_plot) - horizontal_padding,
            max(departure_max_plot, horizontal_padding) + horizontal_padding,
        ]

        fig_departure_tvd = go.Figure()

        fig_departure_tvd.add_trace(
            go.Scatter(
                x=horizontal_departure,
                y=-z_tvd,
                mode="lines",
                line=dict(width=3, color="#009E73"),
                name="Horizontal Departure vs TVD",
            )
        )

        add_marker_departure_tvd(
            fig_departure_tvd,
            fish_top_point,
            "Fish Top",
            "red",
            unit,
        )

        fig_departure_tvd.update_layout(
            title="Vertical Section - Horizontal Departure vs TVD",
            xaxis=dict(
                title=f"Horizontal Departure ({unit})",
                range=departure_x_range,
                tickformat=",.0f",
                separatethousands=True,
            ),
            yaxis=dict(
                title=f"TVD ({unit})",
                tickformat=",.0f",
                separatethousands=True,
            ),
            height=430,
            margin=dict(l=10, r=10, b=32, t=42),
        )

        st.plotly_chart(fig_departure_tvd, use_container_width=True)

        st.markdown("## Impact Simulation Report")
        st.caption(
            "This section recreates the report structure from the reference document using the current simulator inputs."
        )

        report_inputs = pd.DataFrame(
            [
                {"Input": "Mud Weight", "Value": f"{format_report_number(mud_weight_ppg, 3)} lb/gal"},
                {
                    "Input": "Friction Coefficient",
                    "Value": format_report_number(friction_coefficient, 3),
                },
                {
                    "Input": "Well Profile",
                    "Value": well_profile,
                },
                *(
                    [
                        {
                            "Input": "CT BHA Type",
                            "Value": ct_operation_type,
                        }
                    ]
                    if is_ct_job
                    else []
                ),
                {
                    "Input": "Hole / Casing ID",
                    "Value": f"{hole_id_in:g} in ({wellbore_description})",
                },
                {
                    "Input": "Work String",
                    "Value": (
                        f"{work_string_description}, OD {format_report_number(work_string_od, 3)} in, "
                        f"{format_report_number(work_string_weight_lbft, 3)} lb/ft"
                    ),
                },
                {
                    "Input": "Applied Firing Load Up",
                    "Value": f"{applied_firing_load_lbf:,.0f} lb",
                },
                *(
                    [
                        {
                            "Input": "Requested Down Firing Load",
                            "Value": f"{applied_setdown_lbf:,.0f} lb",
                        },
                        {
                            "Input": "CT Jar Push Limit",
                            "Value": (
                                f"{jar_push_limit_lbf:,.0f} lb"
                                if jar_push_limit_lbf is not None
                                else "Manual / not available"
                            ),
                        },
                        {
                            "Input": "Available Setdown at Jar",
                            "Value": f"{ct_setdown_available_lbf:,.0f} lb",
                        },
                    ]
                    if is_ct_job
                    else []
                ),
                {
                    "Input": "Selected Jar Pull Limit",
                    "Value": (
                        f"{jar_pull_limit_lbf:,.0f} lb"
                        if jar_pull_limit_lbf is not None
                        else "Manual / not available"
                    ),
                },
                {
                    "Input": "Weight Bar Joints in Current BHA" if is_ct_job else "Max DC Joints above Jar",
                    "Value": f"{max_joints_above_jar_work}",
                },
            ]
        )

        client_info = pd.DataFrame(
            [
                {"Field": "Well Name", "Value": well_name},
                {"Field": "Customer", "Value": customer},
                {"Field": "Customer Rep", "Value": customer_representative},
                {"Field": "Customer Rep Email", "Value": customer_email},
                {"Field": "Prepared by", "Value": prepared_by},
            ]
        )

        st.markdown("### Customer / Simulation Header")
        compact_dataframe(client_info, height=210, width_ratio=0.58)

        compact_dataframe(report_inputs, height=210, width_ratio=0.58)

        ct_buckling_display = None
        if is_ct_job:
            ct_buckling_display = pd.DataFrame(
                [
                    {"Parameter": key, "Value": value}
                    for key, value in ct_buckling_details.items()
                ]
            )
            st.markdown("### CT Buckling / Setdown Limit")
            compact_dataframe(
                ct_buckling_display,
                height=260,
                decimal_cols={"Value": 3},
                width_ratio=0.58,
            )

        impact_display_cols = [
            "Number of Joints above Jar",
            "Impact (lb)",
            "Impulse (lb-sec)",
            "Up Impact Ratio",
        ]
        if is_ct_job:
            impact_display = impact_by_joints[
                impact_by_joints["Number of Joints above Jar"] == max_joints_above_jar_work
            ][impact_display_cols].copy()
        else:
            impact_display = impact_by_joints[impact_display_cols].copy()
        if is_ct_job and impact_by_joints_down is not None:
            down_cols = [
                "Number of Joints above Jar",
                "Down Impact (lb)",
                "Down Impulse (lb-sec)",
                "Down Impact Ratio",
            ]
            impact_display = impact_display.merge(
                impact_by_joints_down[down_cols],
                on="Number of Joints above Jar",
                how="left",
            )
        impact_display["Impact (lb)"] = impact_display["Impact (lb)"].round(0)
        impact_display["Impulse (lb-sec)"] = impact_display["Impulse (lb-sec)"].round(0)
        impact_display["Up Impact Ratio"] = impact_display["Up Impact Ratio"].round(2)
        if is_ct_job:
            impact_display["Down Impact (lb)"] = impact_display["Down Impact (lb)"].round(0)
            impact_display["Down Impulse (lb-sec)"] = impact_display[
                "Down Impulse (lb-sec)"
            ].round(0)
            impact_display["Down Impact Ratio"] = impact_display[
                "Down Impact Ratio"
            ].round(2)
            impact_display = impact_display.rename(
                columns={"Number of Joints above Jar": "Weight Bar Joints in Current BHA"}
            )

        st.markdown("### Calculation Results")
        ratio_decimals = {"Up Impact Ratio": 2}
        if is_ct_job:
            ratio_decimals["Down Impact Ratio"] = 2
        st.dataframe(
            format_display_dataframe(impact_display, ratio_decimals),
            use_container_width=True,
            height=330,
        )

        fig_impact_joints = None
        fig_down_joints = None
        show_joint_sensitivity_chart = not is_ct_job
        if show_joint_sensitivity_chart:
            fig_impact_joints = go.Figure()
            fig_impact_joints.add_trace(
                go.Scatter(
                    x=impact_by_joints["Number of Joints above Jar"],
                    y=impact_by_joints["Impact (lb)"],
                    mode="lines+markers",
                    name="Up Impact",
                    line=dict(color=TASMAN_BLUE, width=3),
                )
            )
            fig_impact_joints.add_trace(
                go.Scatter(
                    x=impact_by_joints["Number of Joints above Jar"],
                    y=impact_by_joints["Impulse (lb-sec)"],
                    mode="lines+markers",
                    name="Up Impulse",
                    line=dict(color=TASMAN_ORANGE, width=3),
                    yaxis="y2",
                )
            )
            fig_impact_joints.update_layout(
                title="Up Jarring - Impact and Impulse by Joints above Jar",
                yaxis=dict(
                    title="Impact (lb)",
                    tickformat=",.0f",
                    separatethousands=True,
                ),
                yaxis2=dict(
                    title="Impulse (lb-sec)",
                    overlaying="y",
                    side="right",
                    tickformat=",.0f",
                    separatethousands=True,
                ),
                xaxis=dict(
                    title="Number of Joints above Jar",
                    tickformat=",.0f",
                    separatethousands=True,
                    dtick=1,
                ),
                height=360,
                margin=dict(l=10, r=10, b=32, t=42),
                legend=dict(orientation="h", y=1.12),
            )
            set_trace_hovertemplates(fig_impact_joints, "Joints")
            st.plotly_chart(fig_impact_joints, use_container_width=True)

        if show_joint_sensitivity_chart and is_ct_job and impact_by_joints_down is not None:
            fig_down_joints = go.Figure()
            fig_down_joints.add_trace(
                go.Scatter(
                    x=impact_by_joints_down["Number of Joints above Jar"],
                    y=impact_by_joints_down["Down Impact (lb)"],
                    mode="lines+markers",
                    name="Down Impact",
                    line=dict(color=TASMAN_BLUE, width=3),
                )
            )
            fig_down_joints.add_trace(
                go.Scatter(
                    x=impact_by_joints_down["Number of Joints above Jar"],
                    y=impact_by_joints_down["Down Impulse (lb-sec)"],
                    mode="lines+markers",
                    name="Down Impulse",
                    line=dict(color=TASMAN_ORANGE, width=3),
                    yaxis="y2",
                )
            )
            fig_down_joints.update_layout(
                title="Down Jarring - Impact and Impulse by Joints above Jar",
                yaxis=dict(
                    title="Impact (lb)",
                    tickformat=",.0f",
                    separatethousands=True,
                ),
                yaxis2=dict(
                    title="Impulse (lb-sec)",
                    overlaying="y",
                    side="right",
                    tickformat=",.0f",
                    separatethousands=True,
                ),
                xaxis=dict(
                    title="Number of Joints above Jar",
                    tickformat=",.0f",
                    separatethousands=True,
                    dtick=1,
                ),
                height=360,
                margin=dict(l=10, r=10, b=32, t=42),
                legend=dict(orientation="h", y=1.12),
            )
            set_trace_hovertemplates(fig_down_joints, "Joints", "Down Impact", "Down Impulse")
            st.plotly_chart(fig_down_joints, use_container_width=True)

        st.markdown("### Applied Load Sweep")
        if is_ct_job:
            st.caption(
                "Up-jarring sweep for the BHA as entered. To evaluate a different weight bar configuration, update the BHA and run again."
            )
        else:
            st.caption(
                "Select a joint count from the general result, then sweep the applied load for that configuration."
            )

        sweep_cols = st.columns(4)
        sweep_1 = sweep_cols[0]
        sweep_2 = sweep_cols[1]
        sweep_3 = sweep_cols[2]
        sweep_4 = sweep_cols[3]
        sweep_limit_lbf = jar_pull_limit_lbf or applied_firing_load_lbf
        sweep_reference_load_lbf = applied_firing_load_lbf
        if is_ct_job:
            sweep_joint_count = max_joints_above_jar_work
            sweep_1.metric("Weight Bar Joints", f"{sweep_joint_count}")
        else:
            sweep_joint_count = sweep_1.selectbox(
                "DC Joints above Jar",
                list(range(0, max_joints_above_jar_work + 1)),
                index=min(3, max_joints_above_jar_work),
            )
        sweep_config_label = (
            "Current BHA" if is_ct_job else f"{sweep_joint_count} DC Joints"
        )
        sweep_start_key = f"sweep_start_lbf_{catalog_application.lower()}_v2"
        sweep_end_key = f"sweep_end_lbf_{catalog_application.lower()}_v2"
        sweep_step_key = f"sweep_step_lbf_{catalog_application.lower()}_v2"
        for sweep_key in [sweep_start_key, sweep_end_key]:
            if (
                sweep_limit_lbf is not None
                and sweep_key in st.session_state
                and st.session_state[sweep_key] > sweep_limit_lbf
            ):
                st.session_state[sweep_key] = sweep_limit_lbf
        if is_ct_job:
            default_sweep_start_lbf = 5000.0 if float(sweep_limit_lbf or 0.0) >= 5000.0 else 0.0
            default_sweep_step_lbf = 1000.0
        else:
            default_sweep_start_lbf = min(20000.0, float(sweep_limit_lbf or 0.0))
            default_sweep_step_lbf = 10000.0
        sweep_start_lbf = sweep_2.number_input(
            "Load Start (lb)",
            value=default_sweep_start_lbf,
            min_value=0.0,
            max_value=sweep_limit_lbf,
            step=default_sweep_step_lbf,
            key=sweep_start_key,
        )
        sweep_end_default = (
            float(sweep_limit_lbf or sweep_reference_load_lbf)
            if is_ct_job
            else min(
                sweep_reference_load_lbf,
                sweep_limit_lbf or sweep_reference_load_lbf,
            )
        )
        sweep_end_lbf = sweep_3.number_input(
            "Load End (lb)",
            value=float(sweep_end_default),
            min_value=0.0,
            max_value=sweep_limit_lbf,
            step=default_sweep_step_lbf,
            key=sweep_end_key,
        )
        sweep_step_lbf = sweep_4.number_input(
            "Load Step (lb)",
            value=default_sweep_step_lbf,
            min_value=1.0,
            step=1000.0,
            key=sweep_step_key,
        )

        def build_load_sweep(load_start, load_end, ratio_col, direction_label):
            sweep_df = calculate_impact_by_pull_load(
                load_start,
                load_end,
                sweep_step_lbf,
                int(sweep_joint_count),
                physics_inputs,
            )
            display_df = sweep_df[
                ["Pull Load (lb)", "Impact (lb)", "Impulse (lb-sec)", "Up Impact Ratio"]
            ].copy()
            if direction_label == "Down":
                display_df = display_df.rename(
                    columns={
                        "Pull Load (lb)": "Setdown Load (lb)",
                        "Impact (lb)": "Down Impact (lb)",
                        "Impulse (lb-sec)": "Down Impulse (lb-sec)",
                        "Up Impact Ratio": ratio_col,
                    }
                )
                impact_col = "Down Impact (lb)"
                impulse_col = "Down Impulse (lb-sec)"
                x_col = "Setdown Load (lb)"
            else:
                display_df = display_df.rename(
                    columns={"Up Impact Ratio": ratio_col}
                )
                impact_col = "Impact (lb)"
                impulse_col = "Impulse (lb-sec)"
                x_col = "Pull Load (lb)"

            display_df[impact_col] = display_df[impact_col].round(0)
            display_df[impulse_col] = display_df[impulse_col].round(0)
            display_df[ratio_col] = display_df[ratio_col].round(2)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=display_df[x_col],
                    y=display_df[impact_col],
                    mode="lines+markers",
                    name=f"{direction_label} Impact",
                    line=dict(color=TASMAN_BLUE, width=3),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=display_df[x_col],
                    y=display_df[impulse_col],
                    mode="lines+markers",
                    name=f"{direction_label} Impulse",
                    line=dict(color=TASMAN_ORANGE, width=3),
                    yaxis="y2",
                )
            )
            fig.update_layout(
                title=None,
                xaxis=dict(
                    title="Applied Load (lb)",
                    tickformat=",.0f",
                    separatethousands=True,
                ),
                yaxis=dict(
                    title="Impact (lb)",
                    tickformat=",.0f",
                    separatethousands=True,
                ),
                yaxis2=dict(
                    title="Impulse (lb-sec)",
                    overlaying="y",
                    side="right",
                    tickformat=",.0f",
                    separatethousands=True,
                ),
                height=360,
                margin=dict(l=10, r=10, b=32, t=20),
                legend=dict(orientation="h", y=1.08),
            )
            set_trace_hovertemplates(fig, "Applied Load", f"{direction_label} Impact", f"{direction_label} Impulse")
            return display_df, fig

        pull_report_items = []

        up_sweep_end_lbf = min(sweep_end_lbf, jar_pull_limit_lbf or sweep_end_lbf)
        up_sweep_start_lbf = min(sweep_start_lbf, up_sweep_end_lbf)
        pull_display, fig_pull = build_load_sweep(
            up_sweep_start_lbf,
            up_sweep_end_lbf,
            "Up Impact Ratio",
            "Up",
        )
        st.markdown("#### Up Jarring Sweep")
        st.dataframe(
            format_display_dataframe(pull_display, {"Up Impact Ratio": 2}),
            use_container_width=True,
            height=260,
        )
        st.plotly_chart(fig_pull, use_container_width=True)
        pull_report_items.append(("Up Jarring Applied Load Sweep", pull_display, fig_pull))

        report_body = ""
        report_body += dataframe_to_report_html(client_info, "Customer / Simulation Header")
        report_body += dataframe_to_report_html(report_inputs, "Input Summary")
        report_body += dataframe_to_report_html(
            fish_work[fish_display_cols],
            "CT Milling Impact Point" if ct_mill_mode else "Fish Configuration",
            {
                "OD (in)": 3,
                "ID (in)": 3,
                "Weight_lbft": 3,
                "Component Weight (lbf)": 0,
            },
        )
        report_body += dataframe_to_report_html(
            bha_work[display_cols],
            "BHA / Tool String Configuration",
            {
                "OD (in)": 3,
                "ID (in)": 3,
                "Weight_lbft": 3,
                "Component Weight (lbf)": 0,
            },
        )
        report_body += dataframe_to_report_html(
            work_string_display,
            "Work String above BHA / Tool String",
            {
                "OD (in)": 3,
                "ID (in)": 3,
                "Weight_lbft": 3,
                "Total Weight (lbf)": 0,
            },
        )
        report_body += dataframe_to_report_html(
            impact_display,
            "Calculation Results",
            {
                "Impact (lb)": 0,
                "Impulse (lb-sec)": 0,
                "Up Impact Ratio": 2,
                "Down Impact (lb)": 0,
                "Down Impulse (lb-sec)": 0,
                "Down Impact Ratio": 2,
            },
        )
        report_body += plotly_report_html(
            fig_departure_tvd,
            "Well Trajectory - True Vertical Section",
        )
        for sweep_title, sweep_display, _ in pull_report_items:
            report_body += dataframe_to_report_html(
                sweep_display,
                f"{sweep_title} - {sweep_config_label}",
                {
                    "Impact (lb)": 0,
                    "Impulse (lb-sec)": 0,
                    "Up Impact Ratio": 2,
                    "Down Impact (lb)": 0,
                    "Down Impulse (lb-sec)": 0,
                    "Down Impact Ratio": 2,
                },
            )
        if fig_impact_joints is not None:
            report_body += plotly_report_html(
                fig_impact_joints,
                "Up Jarring - Impact and Impulse by Joints above Jar",
            )
        if fig_down_joints is not None:
            report_body += plotly_report_html(
                fig_down_joints,
                "Down Jarring - Impact and Impulse by Joints above Jar",
            )
        for sweep_title, _, sweep_fig in pull_report_items:
            report_body += plotly_report_html(
                sweep_fig,
                f"{sweep_title} - {sweep_config_label}",
            )

        report_html = build_impact_report_html(
            {
                "meta": f"Well: {well_name or 'N/A'} | Customer: {customer or 'N/A'} | Prepared by: {prepared_by or 'N/A'}",
                "body": report_body,
            }
        )

        st.download_button(
            "Download Impact Report (HTML)",
            report_html,
            file_name="jar_impact_simulation_report.html",
            mime="text/html",
        )

        st.markdown("## Survey Summary")

        ss1, ss2, ss3, ss4 = st.columns(4)

        ss1.metric(f"Final MD ({unit})", f"{from_ft(trajectory['MD'].max(), unit):,.0f}")
        ss2.metric(f"Final TVD ({unit})", f"{from_ft(trajectory['TVD'].max(), unit):,.0f}")
        ss3.metric("Max Inclination", f"{trajectory['Inclination'].max():.1f}°")
        ss4.metric("Max DLS", f"{trajectory['DLS'].max():.2f} °/100ft")

    except Exception as e:
        st.error(f"Error processing survey: {e}")

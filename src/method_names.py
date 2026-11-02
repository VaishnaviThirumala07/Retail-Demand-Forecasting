"""Forecast columns reported by the evaluation, dashboard and API, with readable names."""

TOP_DOWN = "TopDownSparse_method-forecast_proportions"
MIN_TRACE = "MinTraceSparse_method-wls_struct"

# Bottom-up equals the base forecast at SKU level, and top-down of a bias-corrected
# model only changes the split of the same seasonal-naive total, so those are omitted.
DISPLAY_METHODS = {
    "SeasonalNaive": "Seasonal naive",
    "LightGBM/BottomUpSparse": "LightGBM bottom-up",
    f"LightGBM/{TOP_DOWN}": "LightGBM top-down",
    f"LightGBM/{MIN_TRACE}": "LightGBM + MinTrace",
    "LightGBM_BC_Global/BottomUpSparse": "Bias-corrected bottom-up (global)",
    "LightGBM_BC_Dept/BottomUpSparse": "Bias-corrected bottom-up (per dept)",
    f"LightGBM_BC_Global/{MIN_TRACE}": "Bias-corrected MinTrace (global)",
    f"LightGBM_BC_Dept/{MIN_TRACE}": "Bias-corrected MinTrace (per dept)",
}

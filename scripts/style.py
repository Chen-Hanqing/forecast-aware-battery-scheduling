"""Shared model labels and colours, so a model looks the same in every chart.

Colours are categorical slots 1-5 of a palette validated for colour-vision separation
(worst adjacent CVD delta-E 9.1, normal-vision 19.6), assigned in a fixed order and never
cycled. Three of them are below 3:1 contrast on a light surface, so charts that use them
also carry direct labels or a legend, never colour alone. "Actual" is drawn in ink, not a
series hue.
"""
MODEL_ORDER = ["seasonal_naive", "gradient_boosting", "lasso_ar", "lear_asinh", "lasso_ar_hourly",
               "gradient_boosting_aligned", "lasso_ar_aligned"]
LABELS = {
    "seasonal_naive": "Seasonal naive",
    "gradient_boosting": "Gradient boosting",
    "lasso_ar": "Lasso-AR",
    "lear_asinh": "LEAR-style (asinh)",
    "lasso_ar_hourly": "Lasso-AR + hour dummies",
    "gradient_boosting_aligned": "Gradient boosting + same-hour lags",
    "lasso_ar_aligned": "Lasso-AR + hour dummies + same-hour lags",
}
COLORS = {
    "seasonal_naive": "#2a78d6",
    "gradient_boosting": "#eb6834",
    "lasso_ar": "#1baf7a",
    "lear_asinh": "#eda100",
    "lasso_ar_hourly": "#e87ba4",
    "gradient_boosting_aligned": "#8b4a1f",
    "lasso_ar_aligned": "#0b6b4a",
}
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e5e4df"

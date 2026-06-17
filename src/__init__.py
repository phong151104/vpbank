# -*- coding: utf-8 -*-
"""VPINS AIA Challenge — gói mô hình hoá (data, features, metrics, models, cv, tune, predict)."""
import os
# Tránh traceback 'wmic' của joblib/loky trên Windows 11 (wmic đã bị gỡ).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

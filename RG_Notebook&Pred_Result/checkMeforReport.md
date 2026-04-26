specify the thought process of finding tree based models fitted well in several counties, but most other counties' are having high RMSE, so the per county model way was invented.

Per-county model selection
instead of one global ensemble weight, pick the best model per county. SARIMAX might be best for 70 counties but STAR might win for 13 high-outage counties. This is essentially stacking.

Add a simple exponential smoothing baseline as a 4th model — even simpler than SARIMAX, often competitive on persistence-dominated data.

For the STAR model:
N_PCA = 20 (According to EDA): old dimension was 35, now the 20 dims could Captures ~90% variance instead of ~92%. Fewer input dimensions means less overfitting risk on 2,161 timesteps, and the STAR weather encoder has fewer parameters to learn. Training will be noticeably faster since the weather conv block processes 20 channels instead of 35.

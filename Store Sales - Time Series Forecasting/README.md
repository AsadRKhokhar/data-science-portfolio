# Store Sales Forecasting with Prophet

This project implements time series forecasting for store sales data using Facebook’s Prophet model. It incorporates holidays and additional regressors to improve forecast accuracy. This is still a work in progress; the full project will include code to run models for all store-family keys and provide comprehensive summary statistics.

---

## Project Overview

- **Dataset:** Kaggle Store Sales Time Series Forecasting competition  
- **Goal:** Predict future sales at the store-family level using historical sales, holidays, and external regressors  
- **Model:** Prophet  
- **Performance:** Summary to be added  

---

## Project Structure

- `1.eda_data_prep.ipynb` — Exploratory data analysis and data cleaning/preparation for modeling  
- `2.data_prep.py` — Modular data preparation pipeline, including feature engineering and data merging  
- `3.prophet_model_run.ipynb` — Model runs comparing Prophet with and without additional regressors and holidays on two grocery keys  
- More to be added! 

---

## Key Findings

- Adding holiday information and external regressors (oil price, transactions, weekend flags) reduces RMSE by ~20% and MAE by ~9% on validation data.  
- Performance varies across store-family keys; further investigation and tuning may improve accuracy for specific segments.

---

## Next Steps

- Develop a full modeling pipeline supporting parallel batch runs  
- Perform detailed key-level performance analysis to identify outliers  
- Generate final test set predictions and prepare submission files  

---

## License

This project is for personal and educational use.

---

## Contact

For questions, reach out at [asadrkhokhar@gmail.com].

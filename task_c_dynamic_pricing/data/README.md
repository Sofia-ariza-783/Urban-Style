# Data

The Kaggle **Retail Sales Dataset** is read from the first of these that exists:

```
task_c_dynamic_pricing/data/retail_sales_dataset.csv
data/retail_sales_dataset.csv          (repository root, shared by all UrbanStyle tasks)
```

Expected columns: `Transaction ID, Date, Customer ID, Gender, Age, Product Category, Quantity, Price per Unit, Total Amount`.

If the file is missing, `prepare_data.py` prints a warning and falls back to a synthetic dataset with the same schema and distributions (written to `synthetic_retail_sales_dataset.csv`), so the whole pipeline still runs.

After adding the real CSV, re-run the pipeline so that models and results reflect the real data:

```bash
python -m data.prepare_data
python -m training.train --algo ppo
python -m training.train --algo dqn
python -m evaluation.evaluate
```

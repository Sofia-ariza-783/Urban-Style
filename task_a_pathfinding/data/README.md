# Data

The Kaggle **Retail Sales Dataset** is read from the first of these that exists:

```
task_a_pathfinding/data/retail_sales_dataset.csv
data/retail_sales_dataset.csv          (repository root, shared by all UrbanStyle tasks)
```

Expected columns: `Transaction ID, Date, Customer ID, Gender, Age, Product Category, Quantity, Price per Unit, Total Amount`.

If the file is missing, `prepare_data.py` prints a warning and falls back to a synthetic dataset with the same schema (written to `synthetic_retail_sales_dataset.csv`), so the whole pipeline still runs.

Only four columns drive the simulation: `Gender`, `Age`, `Product Category` and `Quantity`. Run `python -m data.prepare_data` to write `results/demand_profile.json` (category share, gender share per category, quantity distribution).

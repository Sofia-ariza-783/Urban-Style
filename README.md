# Urban-Style

UrbanStyle is a fictional fashion retailer used as the case study for a university project on applied artificial intelligence. Each task tackles one business question with a different AI technique and shares the same retail sales dataset (`data/retail_sales_dataset.csv`, the Kaggle *Retail Sales Dataset*).

| Task | Question | Techniques | Folder |
|---|---|---|---|
| A | How do customers move through the store, and where does it get crowded? | Grid pathfinding (BFS, A*, congestion-aware Dijkstra), agent simulation, heatmaps, Pygame | [`task_a_pathfinding/`](task_a_pathfinding/) |
| C | How should prices change over time to maximize revenue? | Reinforcement learning (PPO, DQN), Gymnasium, Pygame | [`task_c_dynamic_pricing/`](task_c_dynamic_pricing/) |

Every task folder is self-contained: it has its own `README.md`, `requirements.txt`, tests and results, and all commands are run from inside that folder.

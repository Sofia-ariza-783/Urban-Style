# UrbanStyle, Task A: Customer Pathfinding & Store Layout

A discrete-time simulation shows how customers move through a UrbanStyle store and where it gets crowded. The store is a grid whose shelves are sections (Women's Wear, Men's Wear, Beauty, Accessories & Tech, Fitting Rooms, Checkout). Customers enter, visit the sections on a shopping list drawn from the real sales data, pay and leave. Their routes are planned with **BFS**, **A\*** and a **congestion-aware Dijkstra**, and the visits are accumulated into a **heatmap** with highlighted hotspots. Three store layouts can be compared, and a **Pygame** app lets you watch the store live.

- Academic report: [`docs/REPORT.md`](docs/REPORT.md)

## Architecture

```
 retail_sales_dataset.csv
          │  data/prepare_data.py   clean → category share, gender share, quantity distribution
          ▼
 results/demand_profile.json  +  customer profiles (Gender, Age, Category, Quantity)
          │
          ▼
 simulation/store_sim.py   StoreSimulation (no graphics)
   customer = sampled transaction → shopping list  W/M/B/T (+F) → C → X
   each tick: dwell or walk one cell · spawn · update heat (cumulative) and traffic (decaying)
          │ plans every leg with                      ▲ layouts
          ▼                                           │
 search/algorithms.py                         store/layout.py  +  config.LAYOUTS (ASCII maps)
   BFS · A* (Manhattan) · congestion-aware Dijkstra   walkable · neighbors · access_cells
          │
          ├────────────────────────────┬────────────────────────────┐
          ▼                            ▼                            ▼
 evaluation/experiments.py      pygame_ui/app.py             tests/
   3 layouts × 3 algorithms       live store view              search, simulation, UI
   × seeds → comparison.csv,      heatmap · paths · stats
   metrics.json, figures/
```

## Quick start

Run all commands from this folder (`task_a_pathfinding/`):

```bash
pip install -r requirements.txt

python -m data.prepare_data              # demand profile from data/retail_sales_dataset.csv
python -m pytest tests                   # layouts, algorithms, simulation, UI controls
python -m evaluation.experiments         # 3 layouts x 3 algorithms x 3 seeds -> results/
python -m pygame_ui.app                  # interactive store view
```

Options: `python -m evaluation.experiments --customers 60 --seeds 0 1 2`. Headless screenshot of the app: `python -m pygame_ui.app --screenshot results/figures/app.png --ticks 120`.

## Layout

| Path | Purpose |
|---|---|
| `config.py` | Paths, simulation constants, sections and colours, the three ASCII layouts |
| `data/prepare_data.py` | Loading, cleaning, demand profile, customer profiles |
| `store/layout.py` | `StoreLayout`: parsing, validation, neighbours, access cells |
| `search/algorithms.py` | `bfs`, `astar`, `dijkstra_congestion`, `ALGORITHMS` |
| `simulation/store_sim.py` | `StoreSimulation`, customers, heat / traffic grids, hotspots |
| `evaluation/experiments.py` | Multi-seed comparison, tables and figures |
| `pygame_ui/` | `app.py` (interactive view) and minimal widgets |
| `tests/` | Layout, search, simulation and UI-control tests |
| `results/` | `demand_profile.json`, `comparison.csv`, `metrics.json`, `figures/` |

## Store layouts

Legend: `#` wall, `.` floor, `E` entrance, `X` exit, `C` checkout counter, `W` Women's Wear, `M` Men's Wear, `B` Beauty & Cosmetics, `T` Accessories & Tech, `F` Fitting Rooms. Everything except `. E X` is blocked; customers use the floor cells next to a shelf to visit its section.

| Layout | Size | Idea |
|---|---|---|
| Compact 8x6 | 8 × 6 | The brief's example: one open room, shelves in three rows, checkout next to the exit |
| Grid aisles | 16 × 10 | Parallel shelf rows with straight corridors in between and one cross aisle |
| Racetrack loop | 16 × 10 | Central island of shelves surrounded by a loop corridor |

The maps are in `config.LAYOUTS`; adding a layout there makes it available everywhere. Layouts are validated on load (one entrance, one exit, every section reachable).

## Pygame app controls

| Control | Action |
|---|---|
| Algorithm | Click to cycle BFS → A* → Congestion-aware (restarts the run) |
| Layout | Click to cycle the layouts (restarts the run) |
| Customers slider | 1–60 customers, applied when the slider is released |
| Speed slider | 1–30 simulation ticks per second |
| Run / Pause, Step | Play, pause or advance one tick |
| Reset | Restart the run (same seed, so the same customers come back) |
| Heatmap | Toggle the visit-count overlay; red outlines mark hotspots |
| Paths | Toggle each customer's remaining route |

Circles are customers, coloured by the section they are heading to. The left panel shows the tick, customers in store / served / total, the average nodes expanded per search, the average path length and the top three hotspots with the section they belong to. When everyone has left, "All customers served" appears and the heatmap stays on screen.

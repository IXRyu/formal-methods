import pandas as pd
import pm4py

# 1. Load the CLEANED data
# We don't need to parse dates again if we just want the sequence, 
# but it's good practice to keep the types consistent.
print("Loading data...")
df = pd.read_csv('cleaned_eventlog.csv')
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

# 2. Filter (Optional but Recommended)
# If the map is too messy, uncomment the line below to keep only the 50 most common paths.
# df = pm4py.filter_variants_top_k(df, 50)

print("Discovering process map (Heuristics Miner)...")

# 3. Discover the Map (Heuristic Net)
# dependency_threshold: 0.5 means "only show paths that happen frequently"
# Increase this (e.g., 0.9) to simplify the map further.
heu_net = pm4py.discover_heuristics_net(
    df, 
    dependency_threshold=0.5, 
    and_threshold=0.65, 
    loop_two_threshold=0.5
)

# 4. Save and View
print("Saving map to 'process_map.png'...")
pm4py.save_vis_heuristics_net(heu_net, 'process_map.png')

# If you have Graphviz configured correctly, this window will pop up:
pm4py.view_heuristics_net(heu_net)

print("Done! Check your folder for 'process_map.png'.")
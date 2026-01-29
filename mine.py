import pandas as pd
import pm4py
import matplotlib.pyplot as plt
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness
from pm4py.algo.evaluation.precision import algorithm as precision
from pm4py.algo.evaluation.simplicity import algorithm as simplicity
from pm4py.algo.evaluation.generalization import algorithm as generalization
from pm4py.objects.log.obj import EventLog
import os

INPUT_XES = "dataset/BPI_Challenge_2012.xes"
OUTPUT_CSV = "eventlog.csv" 
OUTPUT_DIR = "bpi_results"
SAMPLE_RATIO = 1.0 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_convert_and_filter(xes_path, csv_out_path):
    """
    1. Loads XES
    2. Converts to DataFrame and saves to CSV (as requested)
    3. Filters for COMPLETE lifecycle to ensure valid process models
    """
    print(f"--- 1. Loading XES file: {xes_path} ---")
    log = pm4py.read_xes(xes_path)
    
    print("--- 2. Converting to DataFrame ---")
    df = pm4py.convert_to_dataframe(log)
    
    print(f"   -> Saving full converted log to {csv_out_path}...")
    df.to_csv(csv_out_path, index=False)
    
    print("--- 3. Preprocessing (Filtering Lifecycle) ---")
    
    if 'lifecycle:transition' in df.columns:
        print("   -> Filtering for 'COMPLETE' lifecycle events only...")
        initial_len = len(df)
        df = df[df['lifecycle:transition'].astype(str).str.upper() == 'COMPLETE']
        print(f"   -> Reduced events from {initial_len} to {len(df)}")
    
    df = df.sort_values(['case:concept:name', 'time:timestamp'])
    
    return df

def get_full_log(df, sample_ratio=1.0):
    """
    Converts DataFrame to a single EventLog object (No Train/Test Split)
    """
    print("\n--- 4. Converting to Full EventLog ---")
    
    if sample_ratio < 1.0:
        print(f"   -> Sampling {sample_ratio*100}% of cases for speed...")
        case_ids = df['case:concept:name'].unique()
        import random
        selected_ids = random.sample(list(case_ids), int(len(case_ids) * sample_ratio))
        df = df[df['case:concept:name'].isin(selected_ids)]

    print(f"   -> Total events: {len(df)}")
    
    # Convert the whole dataframe to a log
    log = pm4py.convert_to_event_log(df)
    
    return log

def plot_metrics(alpha_metrics, heuristic_metrics, inductive_metrics, img_name):
    algorithms = ['Alpha', 'Heuristic', 'Inductive']
    metrics_names = ['Fitness', 'Precision', 'Simplicity', 'Generalization']
    
    data = {
        'Alpha': alpha_metrics,
        'Heuristic': heuristic_metrics,
        'Inductive': inductive_metrics
    }
    
    df_plot = pd.DataFrame(data, index=metrics_names).T
    print(f"\nMetrics Table:\n{df_plot}")
    
    ax = df_plot.plot(kind='bar', figsize=(12, 6), width=0.8)
    plt.title("BPI 2012 Results (Full Dataset)", fontsize=16)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
        
    plt.tight_layout()
    plt.savefig(img_name)
    plt.close()
    print(f"Plot saved to {img_name}")


def alpha_mining(log_train, log_test, img_name, **params):
    print("\n----- Alpha Miner -----")
    try:
        net, im, fm = alpha_miner.apply(log_train)
        gviz = pn_visualizer.apply(net, im, fm)
        pn_visualizer.save(gviz, img_name)
        
        fit = fitness.apply(log_test, net, im, fm, variant=params.get('variant'))["average_trace_fitness"]
        prec = precision.apply(log_test, net, im, fm)
        simp = simplicity.apply(net)
        gen = generalization.apply(log_test, net, im, fm)
        return [fit, prec, simp, gen]
    except Exception as e:
        print(f"Alpha Miner failed: {e}")
        return [0, 0, 0, 0]

def heuristic_mining(log_train, log_test, img_name, **params):
    print("\n----- Heuristic Miner -----")
    try:
        net, im, fm = heuristics_miner.apply(log_train, parameters={
            heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: params.get('dependency_threshold', 0.5),
            heuristics_miner.Variants.CLASSIC.value.Parameters.AND_MEASURE_THRESH: params.get('and_threshold', 0.65)
        })
        gviz = pn_visualizer.apply(net, im, fm)
        pn_visualizer.save(gviz, img_name)
        
        fit = fitness.apply(log_test, net, im, fm, variant=params.get('variant'))["average_trace_fitness"]
        prec = precision.apply(log_test, net, im, fm)
        simp = simplicity.apply(net)
        gen = generalization.apply(log_test, net, im, fm)
        return [fit, prec, simp, gen]
    except Exception as e:
        print(f"Heuristic Miner failed: {e}")
        return [0, 0, 0, 0]

def inductive_mining(log_train, log_test, img_name, **params):
    print(f"\n----- Inductive Miner (Noise: {params.get('noise_threshold')}) -----")
    try:
        net, im, fm = pm4py.discover_petri_net_inductive(log_train, noise_threshold=params.get('noise_threshold', 0.2))
        gviz = pn_visualizer.apply(net, im, fm)
        pn_visualizer.save(gviz, img_name)
        
        fit = fitness.apply(log_test, net, im, fm, variant=params.get('variant'))["average_trace_fitness"]
        prec = precision.apply(log_test, net, im, fm)
        simp = simplicity.apply(net)
        gen = generalization.apply(log_test, net, im, fm)
        return [fit, prec, simp, gen]
    except Exception as e:
        print(f"Inductive Miner failed: {e}")
        return [0, 0, 0, 0]


def main():
    try:
        df = load_convert_and_filter(INPUT_XES, OUTPUT_CSV)
    except FileNotFoundError:
        print(f"Error: {INPUT_XES} not found.")
        return
    except Exception as e:
        print(f"Error loading/converting data: {e}")
        return

    # Use single function to get full log
    log = get_full_log(df, sample_ratio=SAMPLE_RATIO)
    
    base_params = {'variant': fitness.Variants.TOKEN_BASED}

    # Pass 'log' as both training and testing data
    alpha_res = alpha_mining(log, log, f'{OUTPUT_DIR}/alpha_net.png', **base_params)

    heu_params = {**base_params, 'dependency_threshold': 0.5, 'and_threshold': 0.65}
    heu_res = heuristic_mining(log, log, f'{OUTPUT_DIR}/heuristic_net.png', **heu_params)

    ind_params = {**base_params, 'noise_threshold': 0.2}
    ind_res = inductive_mining(log, log, f'{OUTPUT_DIR}/inductive_net_0.2.png', **ind_params)

    plot_metrics(alpha_res, heu_res, ind_res, f'{OUTPUT_DIR}/bpi2012_comparison.png')
    
    print("\n--- Running Inductive Noise Sensitivity Loop ---")
    for n in [0.0, 0.4, 0.8]:
        p = {**base_params, 'noise_threshold': n}
        ind_loop_res = inductive_mining(log, log, f'{OUTPUT_DIR}/inductive_net_{n}.png', **p)
        plot_metrics(alpha_res, heu_res, ind_loop_res, f'{OUTPUT_DIR}/bpi2012_comparison_noise_{n}.png')

if __name__ == "__main__":
    main()
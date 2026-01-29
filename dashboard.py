import streamlit as st
import pandas as pd
import pm4py
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ OPENAI_API_KEY environment variable is not set. Please check your .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

st.set_page_config(page_title="BPI 2012 Mining Dashboard", layout="wide", page_icon="📊")

# Custom CSS for better chat styling
st.markdown("""
<style>
    /* Chat container styling */
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        max-height: 500px;
        overflow-y: auto;
    }
    
    /* Message bubbles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background: white;
        color: #333;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Chat header */
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 15px 15px 0 0;
        margin-bottom: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .chat-header h3 {
        margin: 0;
        font-size: 1.1rem;
    }
    
    /* Chat body */
    .chat-body {
        background-color: #f0f2f5;
        padding: 20px;
        border-radius: 0 0 15px 15px;
        min-height: 400px;
        max-height: 450px;
        overflow-y: auto;
    }
    
    /* Scrollbar styling */
    .chat-body::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-body::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 3px;
    }
    
    .chat-body::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 3px;
    }
    
    .chat-body::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        gap: 4px;
        padding: 12px 18px;
        background: white;
        border-radius: 18px;
        width: fit-content;
    }
    
    .typing-indicator span {
        width: 8px;
        height: 8px;
        background: #667eea;
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out;
    }
    
    .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
    .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    /* Input area styling */
    .stChatInput {
        border-radius: 25px !important;
    }
    
    .stChatInput > div {
        border-radius: 25px !important;
    }
    
    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    
    /* Status badges */
    .status-online {
        width: 10px;
        height: 10px;
        background: #4CAF50;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
    }
    
    /* Improve overall tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

RESULTS_DIR = "bpi_results"
DATA_FILE = "eventlog.csv"

@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='mixed')
    df['case:concept:name'] = df['case:concept:name'].astype(str)
    
    df = df.sort_values(['case:concept:name', 'time:timestamp'])
    df['next_act'] = df.groupby('case:concept:name')['concept:name'].shift(-1)
    df['next_time'] = df.groupby('case:concept:name')['time:timestamp'].shift(-1)
    df['duration_hours'] = (df['next_time'] - df['time:timestamp']).dt.total_seconds() / 3600
    return df

def generate_context_string(df, log):
    n_cases = len(df['case:concept:name'].unique())
    n_events = len(df)
    variants = pm4py.get_variants(log)
    n_variants = len(variants)

    bottlenecks = df.groupby(['concept:name', 'next_act'])['duration_hours'].mean().sort_values(ascending=False).head(5)
    
    context = f"""
    You are a Senior Process Mining Analyst for the BPI 2012 Financial Loan Process.
    The user has already mined the data using Alpha, Heuristic, and Inductive miners.
    
    DATA METRICS:
    - Total Cases: {n_cases}
    - Total Events: {n_events}
    - Distinct Process Variants: {n_variants}
    
    TOP 5 BOTTLENECKS (Avg Transition Time in Hours):
    {bottlenecks.to_string()}
    
    DOMAIN KNOWLEDGE (BPI 2012):
    - This is a loan application process for a Dutch financial institute.
    - 'A_' events are automated system steps.
    - 'W_' events are manual work items (calls, validations).
    - 'O_' events represent offers sent to customers.
    - 'W_Completeren aanvraag' is typically the longest waiting stage.
    """
    return context

st.title("BPI 2012 Dashboard")
df = load_data(DATA_FILE)

if df is None:
    st.error(f"❌ Could not find `{DATA_FILE}`. Please run your `mine.py` script first to generate the data.")
    st.stop()

log = pm4py.convert_to_event_log(df)
process_context = generate_context_string(df, log)

tab1, tab2, tab3 = st.tabs(["🔍 Model Results", "⚠️ Bottleneck Analysis", "🧠 AI Analyst & Prediction"])

with tab1:
    st.header("Process Discovery Results")
    st.info("Visualizing pre-computed Petri nets from the offline mining pipeline.")
    
    col_nav, col_img = st.columns([1, 3])
    
    with col_nav:
        st.subheader("Select Model")
        model_type = st.radio(
            "Algorithm:",
            ("Heuristic Miner", "Inductive Miner (Noise 0.2)", "Alpha Miner", "Metric Comparison")
        )
        
        st.markdown("---")
        st.markdown("**Model Insights:**")
        if model_type == "Alpha Miner":
            st.warning("Alpha miner often produces 'spaghetti' models on BPI data due to complex loops.")
        elif model_type == "Heuristic Miner":
            st.success("Heuristic miner is robust to noise. Showing dependency threshold 0.5.")
        elif "Inductive" in model_type:
            st.success("Inductive miner guarantees a sound process tree. Noise threshold 0.2 used.")

    with col_img:
        image_path = None
        caption = ""
        
        if model_type == "Heuristic Miner":
            image_path = f"{RESULTS_DIR}/heuristic_net.png"
            caption = "Heuristic Miner Result"
        elif model_type == "Inductive Miner (Noise 0.2)":
            image_path = f"{RESULTS_DIR}/inductive_net_0.2.png"
            caption = "Inductive Miner Result (Noise 0.2)"
        elif model_type == "Alpha Miner":
            image_path = f"{RESULTS_DIR}/alpha_net.png"
            caption = "Alpha Miner Result"
        elif model_type == "Metric Comparison":
            image_path = f"{RESULTS_DIR}/bpi2012_comparison.png"
            caption = "Fitness, Precision, Simplicity, Generalization Comparison"

        if image_path and os.path.exists(image_path):
            st.image(image_path, caption=caption, use_container_width=True)
        else:
            st.warning(f"⚠️ Image not found at `{image_path}`. Did the mining script finish successfully?")

with tab2:
    st.header("Performance & Bottleneck Detection")
    st.markdown("Directly calculated from the dataset event timestamps.")
    
    col_vis, col_stats = st.columns([3, 1])
    
    with col_vis:
        with st.spinner("Analyzing process flow for bottlenecks..."):
            metric_choice = st.selectbox("Metric:", ["Mean Duration", "Total Duration", "Frequency"])
            
            agg = "mean" if metric_choice == "Mean Duration" else "sum"
            if metric_choice == "Frequency":
                dfg, start, end = pm4py.discover_dfg(log)
                pm4py.save_vis_dfg(dfg, start, end, "temp_vis.png")
            else:
                dfg, start, end = pm4py.discover_dfg(log)
                pm4py.save_vis_performance_dfg(dfg, start, end, "temp_vis.png", metric=agg)
                
            st.image("temp_vis.png", caption=f"Process Map ({metric_choice})", use_container_width=True)

    with col_stats:
        st.subheader("Slowest Transitions")
        slowest = df.groupby(['concept:name', 'next_act'])['duration_hours'].mean().sort_values(ascending=False).head(10)
        st.dataframe(slowest.rename("Avg Hours"))

with tab3:
    col_chat, col_pred = st.columns([1.2, 1])
    
    with col_chat:
        # Chat header with online status
        st.markdown("""
        <div class="chat-header">
            <span class="status-online"></span>
            <h3>🤖 AI Process Analyst</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize chat messages
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "system", "content": process_context}, 
                {"role": "assistant", "content": "👋 Hello! I've analyzed the BPI 2012 mining results. I can help you understand:\n\n• **Bottlenecks** - Why certain transitions are slow\n• **Algorithm comparison** - Alpha vs Heuristic vs Inductive\n• **Process optimization** - Recommendations for improvement\n\nWhat would you like to explore?"}
            ]

        # Chat messages container
        chat_container = st.container(height=420)
        
        with chat_container:
            for msg in st.session_state.messages:
                if msg["role"] == "system":
                    continue
                elif msg["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(msg["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("💬 Ask about the process analysis...", key="chat_input"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Thinking..."):
                        stream = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                            stream=True,
                        )
                        response = st.write_stream(stream)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    with col_pred:
        st.markdown("""
        <div class="chat-header" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <h3>🔮 Next Best Action Predictor</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        st.info("Select a case from the database to predict the next activity in the process flow.")
        
        # Case selection with search
        case_id = st.selectbox(
            "📂 Select Case ID:",
            df['case:concept:name'].unique()[:50],
            help="Choose a case to analyze its trace and predict the next step"
        )
        
        trace_df = df[df['case:concept:name'] == case_id].sort_values('time:timestamp')
        full_trace = trace_df['concept:name'].tolist()
        
        if len(full_trace) > 1:
            # Trace visualization
            st.markdown("---")
            
            # Timeline slider
            cutoff = st.slider(
                "⏪ Rewind trace to step:",
                1, len(full_trace)-1, len(full_trace)-1,
                help="Move the slider to go back in the process history"
            )
            
            history = full_trace[:cutoff]
            actual_next = full_trace[cutoff]
            
            # Display trace as a nice flow
            st.markdown("**📜 Current Process History:**")
            
            # Create a visual trace representation
            trace_html = ""
            for i, event in enumerate(history):
                if i > 0:
                    trace_html += " → "
                # Color code by event type
                if event.startswith("A_"):
                    color = "#667eea"
                elif event.startswith("W_"):
                    color = "#f093fb"
                elif event.startswith("O_"):
                    color = "#4facfe"
                else:
                    color = "#43e97b"
                trace_html += f'<span style="background-color: {color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; white-space: nowrap;">{event}</span>'
            
            st.markdown(f'<div style="background: #f8f9fa; padding: 15px; border-radius: 10px; overflow-x: auto; white-space: nowrap;">{trace_html}</div>', unsafe_allow_html=True)
            
            st.markdown("")
            
            # Prediction button with better styling
            col_btn, col_info = st.columns([1, 1])
            
            with col_btn:
                predict_clicked = st.button("🎯 Predict Next Step", use_container_width=True, type="primary")
            
            with col_info:
                st.caption(f"Step {cutoff} of {len(full_trace)}")
            
            if predict_clicked:
                with st.spinner("🔮 Analyzing patterns..."):
                    pred_prompt = f"""
                    You are a predictive process monitoring AI.
                    
                    PROCESS CONTEXT:
                    - Dataset: BPI Challenge 2012 (Financial Loan Application)
                    - History of events for this case: {history}
                    
                    TASK:
                    Based on standard loan application flows (Application -> Validation -> Offer -> Decision),
                    predict the single most likely NEXT event.
                    
                    FORMAT:
                    Prediction: [Event Name]
                    Confidence: [High/Medium/Low]
                    Reasoning: [1 sentence explanation]
                    """
                    
                    try:
                        completion = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a Process Mining expert predictor."},
                                {"role": "user", "content": pred_prompt}
                            ]
                        )
                        result = completion.choices[0].message.content
                        
                        # Display prediction in a nice card
                        st.markdown("---")
                        st.markdown("### 📊 Prediction Result")
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; padding: 20px; border-radius: 15px; margin: 10px 0;">
                            {result.replace(chr(10), '<br>')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show actual vs predicted
                        st.markdown("")
                        if actual_next.lower() in result.lower():
                            st.success(f"✅ **Actual Next Event:** `{actual_next}` — **Match!**")
                            st.balloons()
                        else:
                            st.warning(f"❌ **Actual Next Event:** `{actual_next}` — Prediction diverged from reality")
                            
                    except Exception as e:
                        st.error(f"⚠️ API Error: {e}")
        else:
            st.warning("⚠️ This case has only one event. Select a case with more events for prediction.")
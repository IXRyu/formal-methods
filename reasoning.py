import pandas as pd
import pm4py
import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

input_file = "eventlog.csv"
output_prompt_file = "llm_prompt.txt"
output_response_file = "llm_response.txt"

def generate_llm_context():
    """
    Generates the context/prompt for LLM analysis based on process mining metrics.
    Returns the prompt string.
    """
    print("--- GENERATING CONTEXT FOR LLM REASONING ---")
    
    try:
        df = pd.read_csv(input_file)
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='mixed')
        df['case:concept:name'] = df['case:concept:name'].astype(str)
        log = pm4py.convert_to_event_log(df)
    except FileNotFoundError:
        print("Error: File not found.")
        return None

    net, im, fm = pm4py.discover_petri_net_heuristics(df, dependency_threshold=0.5)
    fitness = pm4py.fitness_token_based_replay(log, net, im, fm)['log_fitness']
    precision = pm4py.precision_token_based_replay(log, net, im, fm)
    
    dfg, start, end = pm4py.discover_dfg(log)
    df_sorted = df.sort_values(['case:concept:name', 'time:timestamp'])
    df_sorted['next_time'] = df_sorted.groupby('case:concept:name')['time:timestamp'].shift(-1)
    df_sorted['next_act'] = df_sorted.groupby('case:concept:name')['concept:name'].shift(-1)
    df_sorted['duration'] = (df_sorted['next_time'] - df_sorted['time:timestamp']).dt.total_seconds() / 3600
    
    edges = df_sorted.groupby(['concept:name', 'next_act'])['duration'].mean().reset_index()
    edges = edges.sort_values('duration', ascending=False).head(3)
    
    prompt = f"""
    I am working on a Process Mining project analyzing the BPI Challenge 2012 (Financial Loan Application Process).
    Please act as a Senior Process Analyst and perform the following tasks:
    1. Generate a Management Report summarizing the process health.
    2. Identify Anomalies and Problems based on the metrics.
    3. Suggest Optimizations to fix the bottlenecks.
    4. Predict what might go wrong if the bottlenecks are not fixed.

    HERE IS THE DATA FROM MY PYTHON ANALYSIS:

    [METRICS]
    - Model Fitness: {fitness:.4f} (Ability to replay observed cases)
    - Model Precision: {precision:.4f} (Ability to disallow bad behavior)
    - Total Cases: {len(log)}
    - Total Events: {len(df)}

    [TOP 3 BOTTLENECKS (Slowest Hand-overs)]
    """
    
    for i, row in edges.iterrows():
        prompt += f"   - From '{row['concept:name']}' to '{row['next_act']}': Takes avg {row['duration']:.2f} hours.\n"

    prompt += """
    [DOMAIN CONTEXT]
    - This is a loan application process.
    - 'A_APPROVED' means the bank approved the loan.
    - 'A_DECLINED' means the bank rejected it.
    - 'W_Completeren aanvraag' is waiting for the customer to add missing info.

    Based on this data, please provide a structured analysis.
    """
    
    print(prompt)
    
    with open(output_prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)
    print("\n" + "="*50)
    print(f"Prompt saved to '{output_prompt_file}'")
    
    return prompt


def send_to_chatgpt(prompt, model="gpt-3.5-turbo"):
    """
    Sends the generated prompt to ChatGPT API and returns the response.
    Requires OPENAI_API_KEY environment variable to be set.
    """
    print("\n--- SENDING PROMPT TO CHATGPT API ---")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it in a .env file or as an environment variable.")
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        print(f"   -> Using model: {model}")
        print("   -> Sending request...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Senior Process Analyst expert in Process Mining, Business Process Management, and data-driven optimization. Provide detailed, structured, and actionable analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        answer = response.choices[0].message.content
        
        print("   -> Response received successfully!")
        return answer
        
    except Exception as e:
        print(f"Error calling ChatGPT API: {e}")
        return None


def save_response(response, prompt):
    """
    Saves the LLM response to a file with metadata.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    output_content = f"""{'='*60}
LLM REASONING ANALYSIS - PROCESS MINING PROJECT
{'='*60}
Generated: {timestamp}
Model: GPT-3.5-turbo (OpenAI ChatGPT)
{'='*60}

{'='*60}
ORIGINAL PROMPT SENT TO LLM:
{'='*60}
{prompt}

{'='*60}
LLM RESPONSE:
{'='*60}
{response}

{'='*60}
END OF ANALYSIS
{'='*60}
"""
    
    with open(output_response_file, "w", encoding="utf-8") as f:
        f.write(output_content)
    
    print(f"\n" + "="*50)
    print(f"Full response saved to '{output_response_file}'")
    return output_response_file


def run_llm_reasoning():
    """
    Main function that orchestrates the entire LLM reasoning pipeline:
    1. Generate context from process mining analysis
    2. Send to ChatGPT API
    3. Save response to file
    """
    print("\n" + "="*60)
    print("  LLM REASONING PIPELINE FOR PROCESS MINING")
    print("="*60 + "\n")
    
    # Step 1: Generate prompt
    prompt = generate_llm_context()
    if not prompt:
        print("Failed to generate context. Exiting.")
        return
    
    # Step 2: Send to ChatGPT
    response = send_to_chatgpt(prompt)
    if not response:
        print("\nFailed to get response from ChatGPT.")
        print("The prompt has been saved to 'llm_prompt.txt'.")
        print("You can manually copy it to ChatGPT or check your API key.")
        return
    
    # Step 3: Save response
    output_file = save_response(response, prompt)
    
    # Print summary
    print("\n" + "="*60)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\n  Prompt file: {output_prompt_file}")
    print(f"  Response file: {output_response_file}")
    print("\n  LLM Response Preview:")
    print("-"*60)
    preview = response[:500] + "..." if len(response) > 500 else response
    print(preview)
    print("-"*60)


if __name__ == "__main__":
    run_llm_reasoning()
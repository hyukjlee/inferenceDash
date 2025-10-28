import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import re
import zipfile
import tempfile
import shutil

# Set page config
st.set_page_config(
    page_title="InferenceMAX Benchmark Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

CENTERED_WIDTH = 1500
DEFAULT_CHART_HEIGHT = 700

def inject_layout_styles():
    st.markdown(
        f"""
        <style>
        .block-container {{
            max-width: {CENTERED_WIDTH}px;
            margin: 0 auto;
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
        }}

        h1, h2, h3 {{
            text-align: center;
        }}

        div[data-testid="stPlotlyChart"] {{
            display: flex;
            justify-content: center;
        }}

        div[data-testid="stPlotlyChart"] > div {{
            max-width: {CENTERED_WIDTH}px;
        }}

        div[data-testid="stDataFrame"] {{
            max-width: {CENTERED_WIDTH}px;
            margin: 0 auto;
        }}

        .custom-divider {{
            height: 1px;
            background: linear-gradient(90deg, rgba(56,189,248,0), rgba(56,189,248,0.6), rgba(56,189,248,0));
            margin: 1.5rem auto;
            max-width: {CENTERED_WIDTH}px;
        }}

        div[data-testid="stMarkdown"] p,
        div[data-testid="stMarkdown"] ul,
        div[data-testid="stMarkdown"] ol {{
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def apply_plot_theme(fig, legend_config=None, height=None, width=None, margin=None):
    """Apply optional layout tweaks while keeping Plotly defaults."""
    updates = {}
    if height is not None:
        updates["height"] = height
    if width is not None:
        updates["width"] = width
    if margin is not None:
        updates["margin"] = margin
    elif height is not None:
        updates["margin"] = dict(l=60, r=60, t=70, b=60)
    
    if updates:
        fig.update_layout(**updates)
    
    if legend_config is not None:
        fig.update_layout(legend=legend_config)
    
    return fig

# Function to parse filename and extract metadata
def parse_filename(filename):
    """Extract metadata from artifact naming patterns."""
    base_name = Path(str(filename)).stem
    metadata = {
        'model': 'unknown',
        'isl_osl': 'unknown',
        'precision': 'unknown',
        'framework': 'unknown',
        'tensor_parallel': '1',
        'concurrency': '1',
        'machine': 'unknown',
        'machine_detail': 'unknown',
        'input_length': 'unknown',
        'output_length': 'unknown',
        'run_id': '0',
    }
    parse_success = False
    
    if not base_name:
        return metadata, parse_success
    
    sanitized = base_name.strip()
    sanitized = re.sub(
        r'^(agg|aggregate|aggregated|metrics|summary|benchmark)_',
        '',
        sanitized,
        flags=re.IGNORECASE,
    )
    
    pattern = re.compile(
        r'''^(?P<model>[^_]+)_(?P<isl_osl>[^_]+)_(?P<precision>[^_]+)_(?P<framework>[^_]+)_tp(?P<tp>\d+)_conc(?P<conc>\d+)(?:_(?P<machine>[^_]+))?(?:_(?P<run>[^_]+))?$''',
        re.IGNORECASE,
    )
    match = pattern.match(sanitized)
    
    if match:
        parse_success = True
        metadata['model'] = match.group('model')
        metadata['isl_osl'] = match.group('isl_osl')
        metadata['precision'] = match.group('precision')
        metadata['framework'] = match.group('framework')
        metadata['tensor_parallel'] = match.group('tp')
        metadata['concurrency'] = match.group('conc')
        metadata['run_id'] = match.group('run') or metadata['run_id']
        
        machine_detail = match.group('machine') or metadata['machine_detail']
        metadata['machine_detail'] = machine_detail
        machine_lower = machine_detail.lower()
        if 'h100' in machine_lower:
            metadata['machine'] = 'H100'
        elif 'h200' in machine_lower:
            metadata['machine'] = 'H200'
        elif 'b200' in machine_lower:
            metadata['machine'] = 'B200'
        elif 'mi300x' in machine_lower:
            metadata['machine'] = 'MI300X'
        elif 'mi325x' in machine_lower:
            metadata['machine'] = 'MI325X'
        elif 'mi355x' in machine_lower:
            metadata['machine'] = 'MI355X'
        else:
            metadata['machine'] = machine_detail
        
        isl_pattern = re.compile(r'(?P<input>\d+k)(?P<output>\d+k?)', re.IGNORECASE)
        isl_match = isl_pattern.match(metadata['isl_osl'])
        if isl_match:
            metadata['input_length'] = isl_match.group('input').lower()
            metadata['output_length'] = isl_match.group('output').lower()
    else:
        try:
            parts = sanitized.split('_')
            if parts:
                metadata['model'] = parts[0] or metadata['model']
            if len(parts) > 1:
                metadata['isl_osl'] = parts[1] or metadata['isl_osl']
            if len(parts) > 2:
                metadata['precision'] = parts[2] or metadata['precision']
            if len(parts) > 3:
                metadata['framework'] = parts[3] or metadata['framework']
            if len(parts) > 4 and parts[4]:
                tp_match = re.search(r'\d+', parts[4])
                metadata['tensor_parallel'] = tp_match.group(0) if tp_match else parts[4]
            if len(parts) > 5 and parts[5]:
                conc_match = re.search(r'\d+', parts[5])
                metadata['concurrency'] = conc_match.group(0) if conc_match else parts[5]
            if len(parts) > 6:
                machine_detail = parts[6]
                metadata['machine_detail'] = machine_detail
                machine_lower = machine_detail.lower()
                if 'h100' in machine_lower:
                    metadata['machine'] = 'H100'
                elif 'h200' in machine_lower:
                    metadata['machine'] = 'H200'
                elif 'b200' in machine_lower:
                    metadata['machine'] = 'B200'
                elif 'mi300x' in machine_lower:
                    metadata['machine'] = 'MI300X'
                elif 'mi325x' in machine_lower:
                    metadata['machine'] = 'MI325X'
                elif 'mi355x' in machine_lower:
                    metadata['machine'] = 'MI355X'
                else:
                    metadata['machine'] = machine_detail
            if len(parts) > 7 and parts[7]:
                metadata['run_id'] = parts[7]
            
            isl_pattern = re.compile(r'(?P<input>\d+k)(?P<output>\d+k?)', re.IGNORECASE)
            isl_match = isl_pattern.match(metadata['isl_osl'])
            if isl_match:
                metadata['input_length'] = isl_match.group('input').lower()
                metadata['output_length'] = isl_match.group('output').lower()
        except Exception as e:
            st.warning(f"Error parsing filename '{filename}': {e}")
    
    return metadata, parse_success

# Function to load all benchmark data
@st.cache_data
def load_benchmark_data():
    """Load all JSON files from zip files in artifacts folder"""
    artifacts_path = Path('./artifacts')
    data_list = []
    
    # Create a temporary directory for extracting zip files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Look for zip files directly in artifacts directory
        zip_files = list(artifacts_path.glob('*.zip'))
        
        if not zip_files:
            #st.warning(f"No zip files found in {artifacts_path}")
            return pd.DataFrame(data_list)
        
        #st.info(f"Found {len(zip_files)} zip file(s) to process")
        
        for zip_file in zip_files:
            try:
                #st.info(f"Processing {zip_file.name}...")
                
                # Extract zip file to temporary directory
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # Create a subfolder for this zip file
                    extract_path = Path(temp_dir) / zip_file.stem
                    extract_path.mkdir(exist_ok=True)
                    zip_ref.extractall(extract_path)
                    
                    # Process JSON files from extracted content
                    json_files = list(extract_path.rglob('*.json'))
                    #st.info(f"Found {len(json_files)} JSON file(s) in {zip_file.name}")
                    
                    for json_file in json_files:
                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                            
                            # Try to parse metadata from the parent folder name or json filename
                            # The folder structure inside zip might be: artifact_name/benchmark_name.json
                            # We need to get the folder name that contains the metadata
                            parent_folder = json_file.parent.name
                            
                            # If parent folder is just the zip stem, try the json filename
                            if parent_folder == zip_file.stem:
                                # Use json filename without extension
                                folder_name = json_file.stem
                            else:
                                folder_name = parent_folder
                            
                            #st.info(f"Parsing filename: {folder_name}")
                            
                            candidate_names = []
                            if zip_file:
                                candidate_names.append(zip_file.stem)
                            candidate_names.extend([
                                folder_name,
                                json_file.stem,
                                json_file.name
                            ])
                            
                            parent = json_file.parent
                            depth = 0
                            while parent and parent.name and depth < 3:
                                candidate_names.append(parent.name)
                                parent = parent.parent
                                depth += 1
                            
                            unique_candidates = []
                            seen_candidates = set()
                            for candidate in candidate_names:
                                candidate_str = str(candidate).strip()
                                if candidate_str and candidate_str not in seen_candidates:
                                    seen_candidates.add(candidate_str)
                                    unique_candidates.append(candidate_str)
                            
                            metadata = {}
                            metadata_parsed = False
                            last_metadata = None
                            
                            for candidate in unique_candidates:
                                current_metadata, parsed = parse_filename(candidate)
                                last_metadata = current_metadata
                                if parsed:
                                    metadata = current_metadata
                                    metadata_parsed = True
                                    break
                            
                            if not metadata_parsed:
                                metadata = last_metadata if last_metadata else parse_filename('')[0]
                                #st.warning(
                                #    f"Could not fully parse metadata from '{folder_name}'. "
                                #    "Some fields may remain as 'unknown'."
                                #)
                            
                            # Extract benchmark metrics from the JSON structure
                            tp_value = data.get('tp')
                            if tp_value is None:
                                tp_value = metadata.get('tensor_parallel', '1')
                            
                            conc_value = data.get('conc')
                            if conc_value is None:
                                conc_value = metadata.get('concurrency', '1')
                            
                            metadata_model = metadata.get('model', 'unknown')
                            data_model = data.get('model')
                            record_model = metadata_model if metadata_model != 'unknown' else data_model
                            
                            record = {
                                **metadata,
                                'model': record_model or 'unknown',
                                'model_full': data_model or record_model or 'unknown',
                                'hw': data.get('hw'),
                                'framework': data.get('framework', metadata.get('framework')),
                                'precision': data.get('precision', metadata.get('precision')),
                                'tp': str(tp_value),
                                'conc': str(conc_value),
                                
                                # Throughput metrics
                                'token_throughput_per_gpu': data.get('tput_per_gpu'),
                                'output_throughput_per_gpu': data.get('output_tput_per_gpu'),
                                'input_throughput_per_gpu': data.get('input_tput_per_gpu'),
                                
                                # Time to First Token (TTFT)
                                'mean_ttft': data.get('mean_ttft'),
                                'median_ttft': data.get('median_ttft'),
                                'std_ttft': data.get('std_ttft'),
                                'p99_ttft': data.get('p99_ttft'),
                                
                                # Time Per Output Token (TPOT)
                                'mean_tpot': data.get('mean_tpot'),
                                'median_tpot': data.get('median_tpot'),
                                'std_tpot': data.get('std_tpot'),
                                'p99_tpot': data.get('p99_tpot'),
                                
                                # Inter-token Latency (ITL)
                                'mean_itl': data.get('mean_itl'),
                                'median_itl': data.get('median_itl'),
                                'std_itl': data.get('std_itl'),
                                'p99_itl': data.get('p99_itl'),
                                
                                # End-to-End Latency (E2EL)
                                'end_to_end_latency': data.get('mean_e2el'),
                                'median_e2el': data.get('median_e2el'),
                                'std_e2el': data.get('std_e2el'),
                                'p99_e2el': data.get('p99_e2el'),
                                
                                # Inter-token Vitality
                                'mean_intvty': data.get('mean_intvty'),
                                'median_intvty': data.get('median_intvty'),
                                
                                'file_path': str(json_file),
                                'source_zip': str(zip_file)
                            }
                            
                            data_list.append(record)
                        except Exception as e:
                            st.warning(f"Error loading JSON {json_file.name}: {e}")
            except Exception as e:
                st.error(f"Error extracting zip file {zip_file.name}: {e}")
        
        #st.success(f"Successfully loaded {len(data_list)} benchmark records")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return pd.DataFrame(data_list)

# Main dashboard
def main():
    inject_layout_styles()
    
    st.title("InferenceMAX Benchmark Dashboard")
    
    # Load data
    df = load_benchmark_data()
    
    if df.empty:
        st.error("No benchmark data found in ./artifacts folder")
        return
    
      
    # Debug: Show column names
    #st.info(f"DataFrame columns: {list(df.columns)}")
    #st.info(f"DataFrame shape: {df.shape}")
    
    # Check if required columns exist
    #required_columns = ['model', 'isl_osl', 'precision', 'framework', 'tp']
    #missing_columns = [col for col in required_columns if col not in df.columns]
    
    #if missing_columns:
    #    st.error(f"Missing required columns: {missing_columns}")
    #    st.dataframe(df.head())
    #    return
    
    st.sidebar.header("Filter Options")
    st.sidebar.caption("Tune the controls to explore specific benchmark configurations.")
    
    # Model selection
    models = sorted(df['model'].unique())
    selected_model = st.sidebar.selectbox("Select Model", models)
    
    # ISL/OSL selection
    isl_osl_options = sorted(df['isl_osl'].unique())
    selected_isl_osl = st.sidebar.selectbox("Select ISL/OSL", isl_osl_options)
    
    # Precision selection
    precisions = sorted(df['precision'].unique())
    selected_precision = st.sidebar.selectbox("Select Precision", precisions)
    
    # TP (Tensor Parallelism) selection (optional filter)
    tp_options = sorted(df['tp'].unique())
    #selected_tps = st.sidebar.selectbox("Select Tensor Parallelism (TP)", tp_options)

    selected_tps = st.sidebar.multiselect(
        "Select Tensor Parallelism (TP)", 
        tp_options, 
        default='2'
    )
    
    
    # Framework selection (optional filter)
    frameworks = sorted(df['framework'].unique())
    selected_frameworks = st.sidebar.multiselect(
        "Select Framework(s)", 
        frameworks, 
        default='vllm'
    )
    
    
    # Filter data based on selections
    filtered_df = df[
        (df['model'] == selected_model) &
        (df['isl_osl'] == selected_isl_osl) &
        (df['precision'] == selected_precision) &
        (df['framework'].isin(selected_frameworks)) &
        (df['tp'].isin(selected_tps))
    ]
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters")
        return       
    
    # Summary of selected configurations
    tp_numeric = pd.to_numeric(filtered_df['tp'], errors='coerce')
    conc_numeric = pd.to_numeric(filtered_df['conc'], errors='coerce')
    
    tp_summary = (
        f"TP{int(tp_numeric.min())} — TP{int(tp_numeric.max())}"
        if tp_numeric.notna().any()
        else "TP Range: —"
    )
    conc_summary = (
        f"{int(conc_numeric.min())} — {int(conc_numeric.max())}"
        if conc_numeric.notna().any()
        else "—"
    )
    machine_summary = ", ".join(sorted(filtered_df['machine'].dropna().unique())) or "—"
      
    st.subheader("Token Throughput per GPU vs End-to-End Latency")
    st.markdown(
        f"Showing benchmarks for **{selected_model.upper()} | {selected_isl_osl} | {selected_precision.upper()}** "
        f"across {len(filtered_df['framework'].unique())} framework(s) and {len(filtered_df)} run(s)."
    )
    
    # Create scatter plot with Plotly
    fig = go.Figure()
    
    # Sort filtered_df by machine and end_to_end_latency for proper line connections
    filtered_df_sorted = filtered_df.sort_values(['machine', 'end_to_end_latency'])
    
    # Group by machine type and add traces
    for machine in filtered_df_sorted['machine'].unique():
        machine_data = filtered_df_sorted[filtered_df_sorted['machine'] == machine].sort_values('end_to_end_latency')
        
        fig.add_trace(go.Scatter(
            x=machine_data['end_to_end_latency'],
            y=machine_data['token_throughput_per_gpu'],
            mode='markers+lines',
            name=machine,
            marker=dict(size=10),
            line=dict(width=2),
            text=machine_data['framework'] + ' (TP' + machine_data['tp'] + ', CON' + machine_data['conc'] + ')',
            textposition="top center",
            hovertemplate=(
                "<b>%{fullData.name}</b><br>" +
                "E2E Latency: %{x:.3f}s<br>" +
                "Throughput/GPU: %{y:.1f}<br>" +
                "Config: %{text}<br>" +
                "<extra></extra>"
            )
        ))
    
    # Calculate axis ranges with some padding
    x_min = filtered_df['end_to_end_latency'].min()
    x_max = filtered_df['end_to_end_latency'].max()
    x_range = x_max - x_min
    x_padding = x_range * 0.1 if x_range else max(abs(x_min) * 0.05, 0.1)
    
    y_min = filtered_df['token_throughput_per_gpu'].min()
    y_max = filtered_df['token_throughput_per_gpu'].max()
    y_range = y_max - y_min
    y_padding = y_range * 0.1 if y_range else max(abs(y_min) * 0.1, 0.5)
    
    chart_height = DEFAULT_CHART_HEIGHT
    scatter_legend = dict(
        title="GPU Type",
        orientation="v",
        yanchor="middle",
        y=0.5,
        xanchor="left",
        x=1.02,
    )
    apply_plot_theme(fig, legend_config=scatter_legend, height=chart_height, width=CENTERED_WIDTH)
    fig.update_layout(hovermode='closest')
    fig.update_xaxes(
        title="End-to-End Latency (seconds)",
        range=[x_min - x_padding, x_max + x_padding],
        gridwidth=1,
        showgrid=True,
        gridcolor="LightGray",
    )
    fig.update_yaxes(
        title="Token Throughput per GPU (tokens/sec)",
        range=[y_min - y_padding, y_max + y_padding],
        gridwidth=1,
        showgrid=True,
        gridcolor="LightGray",
    )
    
    st.plotly_chart(fig, use_container_width=False)
    
    # Additional scatter: Token Throughput per GPU vs Interactivity
    st.subheader("Token Throughput per GPU vs Interactivity")
    st.markdown(
        f"Showing benchmarks for **{selected_model.upper()} | {selected_isl_osl} | {selected_precision.upper()}** "
        f"across {len(filtered_df['framework'].unique())} framework(s) and {len(filtered_df)} run(s)."
    )
    
    scatter_df = filtered_df.copy()
    scatter_df['conc_numeric'] = pd.to_numeric(scatter_df['conc'], errors='coerce')
    scatter_df['interactivity'] = scatter_df['token_throughput_per_gpu'] / scatter_df['conc_numeric']
    scatter_df = scatter_df[scatter_df['interactivity'].notna()]
    
    fig_interactivity = go.Figure()
    for machine in scatter_df['machine'].unique():
        subset = scatter_df[scatter_df['machine'] == machine]
        fig_interactivity.add_trace(go.Scatter(
            x=subset['interactivity'],
            y=subset['token_throughput_per_gpu'],
            mode='markers+lines',
            line=dict(width=2),
            name=machine,
            marker=dict(
                size=12,
                line=dict(width=1, color='DarkSlateGrey'),
            ),
            text=subset['framework'] + ' (TP' + subset['tp'] + ', CON' + subset['conc'] + ')',
            hovertemplate=(
                "<b>%{fullData.name}</b><br>" +
                "Interactivity: %{x:.1f} tok/s/user<br>" +
                "Throughput/GPU: %{y:.1f} tok/s<br>" +
                "Config: %{text}<br>" +
                "<extra></extra>"
            )
        ))
    
    interactivity_height = DEFAULT_CHART_HEIGHT 
    interactivity_legend = dict(
        title="GPU Type",
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-0.2,
    )
    apply_plot_theme(
        fig_interactivity,
        legend_config=interactivity_legend,
        height=interactivity_height,
        width=CENTERED_WIDTH
    )
    fig_interactivity.update_xaxes(
        title="Interactivity (tokens/sec/user)",
        showgrid=True,
        gridcolor="LightGray",
        gridwidth=1,
    )
    fig_interactivity.update_yaxes(
        title="Token Throughput per GPU (tokens/sec)",
        showgrid=True,
        gridcolor="LightGray",
        gridwidth=1,
    )
    
    st.plotly_chart(fig_interactivity, use_container_width=False)
    
    # Detailed data table
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    st.subheader("Detailed Results")
    st.markdown("Sorted by GPU, framework, tensor parallelism, and concurrency for quick comparisons.")
    
    display_columns = [
        'machine', 'framework', 'tp', 'conc',
        'end_to_end_latency', 'token_throughput_per_gpu',
        'mean_ttft', 'mean_tpot', 'mean_itl'
    ]
    
    # Sort by machine, framework, tp (as int), and concurrency (as int)
    filtered_df['tp_int'] = filtered_df['tp'].astype(int)
    filtered_df['conc_int'] = filtered_df['conc'].astype(int)
    
    display_df = filtered_df[display_columns + ['tp_int', 'conc_int']].sort_values(
        ['machine', 'framework', 'tp_int', 'conc_int']
    ).reset_index(drop=True)
    
    # Drop the helper columns
    display_df = display_df[display_columns]
    
    # Rename columns for display
    display_df.columns = [
        'Machine', 'Framework', 'TP', 'Concurrency',
        'E2E Latency (s)', 'Throughput/GPU',
        'Mean TTFT (s)', 'Mean TPOT (s)', 'Mean ITL (s)'
    ]
    
    # Format numbers
    display_df['E2E Latency (s)'] = display_df['E2E Latency (s)'].apply(lambda x: f"{x:.3f}")
    display_df['Throughput/GPU'] = display_df['Throughput/GPU'].apply(lambda x: f"{x:.1f}")
    display_df['Mean TTFT (s)'] = display_df['Mean TTFT (s)'].apply(lambda x: f"{x:.3f}")
    display_df['Mean TPOT (s)'] = display_df['Mean TPOT (s)'].apply(lambda x: f"{x:.5f}")
    display_df['Mean ITL (s)'] = display_df['Mean ITL (s)'].apply(lambda x: f"{x:.5f}")
    
    st.dataframe(display_df, use_container_width=800)

if __name__ == "__main__":
    main()

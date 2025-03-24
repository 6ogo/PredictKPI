"""
UI components for KPI Predictor application
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Tuple, Optional, Union

from utils import logger, DIALOG_VALUES, SYFTE_VALUES, PRODUKT_VALUES, categorize_age
from utils import import_subjects_from_csv, export_results_to_csv, batch_predict

# --- Enhanced UI for Input ---
def create_enhanced_input_ui():
    """Create an enhanced UI for subject line and preheader input with real-time feedback"""
    st.subheader("Email Content")
    
    # Subject line with character count and real-time feedback
    subject_col, feedback_col = st.columns([3, 1])
    
    with subject_col:
        subject_line = st.text_input(
            'Subject Line',
            key='subject_input',
            help="Enter the email subject line"
        )
        
        # Character and word count
        if subject_line:
            char_count = len(subject_line)
            word_count = len(subject_line.split())
            st.caption(f"Character count: {char_count} | Word count: {word_count}")
    
    with feedback_col:
        # Visual feedback on subject line
        if subject_line:
            # Check length (optimal is usually 30-60 characters)
            if 30 <= char_count <= 60:
                st.success("Length: Good")
            elif char_count > 60:
                st.warning("Length: Long")
            else:
                st.info("Length: Short")
                
            # Check for spam triggers like ALL CAPS, excessive punctuation
            caps_ratio = sum(1 for c in subject_line if c.isupper()) / len(subject_line)
            if caps_ratio > 0.5:
                st.warning("High caps usage")
                
            if '!' in subject_line and '?' in subject_line:
                st.warning("Mixed '!' and '?'")
            elif subject_line.count('!') > 1:
                st.warning("Multiple '!'")
    
    # Preheader with feedback
    preheader = st.text_input(
        'Preheader',
        help="Text that appears after the subject line in inbox previews"
    )
    
    if preheader:
        # Character and word count
        preheader_char_count = len(preheader)
        preheader_word_count = len(preheader.split())
        st.caption(f"Character count: {preheader_char_count} | Word count: {preheader_word_count}")
        
        # Optimal preheader length is usually 50-100 characters
        if preheader_char_count < 40:
            st.info("Preheader is relatively short")
        elif preheader_char_count > 100:
            st.info("Preheader is relatively long")
    
    # Advanced options with GenAI integration
    with st.expander("Advanced Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            use_genai = st.checkbox('Generate alternatives with AI', value=False)
            use_formatting = st.checkbox('Format suggestions with emojis', value=True,
                                       help="Add relevant emojis to the generated subject lines")
        
        with col2:
            focus_options = st.radio(
                "Optimization focus",
                ["Balanced", "Click-through", "Open rate", "Avoid unsubscribes"],
                horizontal=True
            )
            
            tone_options = st.select_slider(
                "Tone",
                options=["Formal", "Professional", "Neutral", "Conversational", "Casual"],
                value="Professional"
            )
    
    return subject_line, preheader, use_genai, {
        'formatting': use_formatting,
        'focus': focus_options,
        'tone': tone_options
    }

# --- Visualization Functions ---
def create_interactive_heatmap(data, metric, title, is_percentage=True, colorscale='Viridis'):
    """
    Create an interactive heatmap using Plotly for a specific metric
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with age groups as index and categories as columns
    metric : str
        The metric name ('Openrate', 'Clickrate', or 'Optoutrate')
    title : str
        The title for the heatmap
    is_percentage : bool
        Whether to format values as percentages
    colorscale : str
        Plotly colorscale to use
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Format data for heatmap
    z = data.values
    x = data.columns
    y = data.index
    
    # Format values for hover text
    if is_percentage:
        hover_text = [[f"{z[i][j]:.2%}" for j in range(len(x))] for i in range(len(y))]
    else:
        hover_text = [[f"{z[i][j]:.4f}" for j in range(len(x))] for i in range(len(y))]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        hoverongaps=False,
        text=hover_text,
        hoverinfo='text+x+y',
        colorscale=colorscale
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Category',
        yaxis_title='Age Group',
        xaxis=dict(
            tickangle=-45,
            side='bottom',
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            autorange='reversed',  # Important to make age groups go from youngest to oldest
            tickfont=dict(size=10)
        ),
        height=400,
        margin=dict(l=50, r=50, t=80, b=100)
    )
    
    return fig

def create_age_heatmap(heatmap_data, metric, title, cmap='viridis', figsize=(12, 6)):
    """
    Create a heatmap for a specific metric by age group
    """
    try:
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get data for this metric
        data = heatmap_data[metric].copy()
        
        # Ensure data is numeric
        data = data.astype(float)
        
        # Format values as percentages
        labels = data.applymap(lambda x: f"{x:.2%}")
        
        # Create heatmap
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        
        # Show all ticks and label them
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_xticklabels(data.columns)
        ax.set_yticklabels(data.index)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                ax.text(j, i, labels.iloc[i, j],
                        ha="center", va="center", 
                        color="white" if data.iloc[i, j] > data.values.mean() else "black")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(f"{title} (%)", rotation=-90, va="bottom")
        
        # Add title and adjust layout
        ax.set_title(title)
        fig.tight_layout()
        
        return fig
    except Exception as e:
        logger.error(f"Error creating heatmap: {str(e)}")
        raise

# --- A/B Testing UI ---
def display_abcd_results(all_options, avg_clickrate, avg_optoutrate, age_data=None):
    """
    Display enhanced A/B/C/D test results with interactive comparisons
    
    Parameters:
    -----------
    all_options : list
        List of tuples (version, subject, preheader, openrate)
    avg_clickrate : float
        Average click rate to display
    avg_optoutrate : float
        Average opt-out rate to display
    age_data : dict, optional
        Age group data for heatmaps
    """
    
    st.subheader("A/B/C/D Test Results")
    
    # Extract data
    versions = [opt[0] for opt in all_options]
    subjects = [opt[1] for opt in all_options]
    preheaders = [opt[2] for opt in all_options]
    openrates = [opt[3] for opt in all_options]
    
    # Find best version
    best_idx = openrates.index(max(openrates))
    current_idx = 0  # Version A is always current
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Version Cards", "Comparison"])
    
    with tab1:
        # Display each version in a card layout
        for i, (version, subject, preheader, openrate) in enumerate(all_options):
            is_current = i == current_idx
            is_best = i == best_idx
            
            # Create card with border
            card_color = "#2f93e0" if is_best else "#f0f2f6"
            card_title = f"Version {version}" + (" (Current)" if is_current else "") + (" ⭐ BEST" if is_best else "")
            
            with st.container():
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 5px; border: 1px solid {card_color}; margin-bottom: 10px;">
                    <h4 style="color: {card_color};">{card_title}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("**Subject:**")
                    st.code(subject, language=None)
                    
                    st.markdown("**Preheader:**")
                    st.code(preheader, language=None)
                
                with col2:
                    # Metrics with comparison to current (version A)
                    delta = None if is_current else openrate - openrates[current_idx]
                    st.metric("Open Rate", f"{openrate:.2%}", f"{delta:.2%}" if delta is not None else None)
                    st.metric("Est. Click Rate", f"{avg_clickrate:.2%}")
                    st.metric("Est. Opt-out Rate", f"{avg_optoutrate:.2%}")
    
    with tab2:
        # Bar chart comparison
        fig = go.Figure()
        
        colors = ['#a2b9bc' if i != best_idx else '#2f93e0' for i in range(len(versions))]
        
        # Add open rate bars
        fig.add_trace(go.Bar(
            x=versions,
            y=openrates,
            text=[f"{rate:.2%}" for rate in openrates],
            textposition='auto',
            marker_color=colors,
            name='Open Rate'
        ))
        
        # Update layout
        fig.update_layout(
            title="Open Rate Comparison",
            xaxis_title="Version",
            yaxis_title="Open Rate",
            yaxis_tickformat=".0%",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a comparison table
        comparison_df = pd.DataFrame({
            'Version': versions,
            'Subject Line': subjects,
            'Preheader': preheaders,
            'Predicted Open Rate': [f"{rate:.2%}" for rate in openrates],
            'Improvement': [f"{(rate - openrates[0]):.2%}" if i > 0 else "-" for i, rate in enumerate(openrates)]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Recommendation summary
        if best_idx != current_idx:
            improvement = openrates[best_idx] - openrates[current_idx]
            st.success(f"🎯 Recommendation: Use Version {versions[best_idx]} for an estimated {improvement:.2%} improvement in open rate")
        else:
            st.info("✅ Your current version (A) is predicted to perform best")

# --- KPI Dashboard ---
def create_kpi_dashboard(delivery_data, threshold=0.5):
    """
    Create an interactive KPI dashboard with key metrics and visualizations
    
    Parameters:
    -----------
    delivery_data : DataFrame
        Delivery data with campaign information
    threshold : float
        Threshold for defining a "good" open rate
    """
    # Calculate key metrics
    avg_open_rate = delivery_data['Openrate'].mean()
    avg_click_rate = delivery_data['Clickrate'].mean()
    avg_optout_rate = delivery_data['Optoutrate'].mean()
    
    total_sendouts = delivery_data['Sendouts'].sum()
    total_opens = delivery_data['Opens'].sum()
    total_clicks = delivery_data['Clicks'].sum()
    
    # Calculate distribution percentiles
    open_rate_percentiles = {
        'low': delivery_data['Openrate'].quantile(0.25),
        'median': delivery_data['Openrate'].quantile(0.5),
        'high': delivery_data['Openrate'].quantile(0.75),
    }
    
    # Create dashboard
    st.subheader("Email Campaign KPI Dashboard")
    
    # Top row - summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Open Rate", f"{avg_open_rate:.2%}")
        st.caption(f"Target: >{threshold:.0%}")
    
    with col2:
        st.metric("Avg Click Rate", f"{avg_click_rate:.2%}")
    
    with col3:
        st.metric("Avg Opt-out Rate", f"{avg_optout_rate:.2%}")
    
    with col4:
        st.metric("Total Sendouts", f"{total_sendouts:,}")
        st.caption(f"Total Opens: {total_opens:,}")
    
    # Performance distribution
    fig = go.Figure()
    
    # Add open rate distribution
    fig.add_trace(go.Histogram(
        x=delivery_data['Openrate'],
        name='Open Rate',
        opacity=0.75,
        marker_color='#1f77b4',
        xbins=dict(
            start=0,
            end=min(1, delivery_data['Openrate'].max() * 1.1),
            size=0.05
        )
    ))
    
    # Add percentile lines
    for label, value in open_rate_percentiles.items():
        fig.add_vline(
            x=value,
            line_dash="dash",
            line_color="#ff7f0e",
            annotation_text=f"{label.title()}: {value:.2%}",
            annotation_position="top right"
        )
    
    # Add target threshold line
    fig.add_vline(
        x=threshold,
        line_dash="solid",
        line_color="#d62728",
        annotation_text=f"Target: {threshold:.0%}",
        annotation_position="top left"
    )
    
    # Update layout
    fig.update_layout(
        title="Open Rate Distribution",
        xaxis_title="Open Rate",
        yaxis_title="Frequency",
        xaxis_tickformat=".0%",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparison by categories
    with st.expander("Performance by Category", expanded=False):
        category_options = ["Dialog", "Syfte", "Product"]
        selected_category = st.selectbox("Select Category", category_options)
        
        # Aggregate data by selected category
        agg_data = delivery_data.groupby(selected_category).agg({
            'Sendouts': 'sum',
            'Opens': 'sum',
            'Clicks': 'sum',
            'Optouts': 'sum'
        }).reset_index()
        
        # Calculate rates
        agg_data['Openrate'] = agg_data['Opens'] / agg_data['Sendouts']
        agg_data['Clickrate'] = agg_data['Clicks'] / agg_data['Sendouts']
        agg_data['Optoutrate'] = agg_data['Optouts'] / agg_data['Sendouts']
        
        # Sort by open rate
        agg_data = agg_data.sort_values('Openrate', ascending=False)
        
        # Create horizontal bar chart
        fig = px.bar(
            agg_data,
            y=selected_category,
            x='Openrate',
            orientation='h',
            color='Openrate',
            color_continuous_scale='blues',
            labels={'Openrate': 'Open Rate'},
            title=f"Open Rate by {selected_category}",
            height=max(400, len(agg_data) * 30)  # Dynamic height based on number of items
        )
        
        # Add target line
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red"
        )
        
        fig.update_layout(xaxis_tickformat=".0%")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data table
        display_cols = [selected_category, 'Sendouts', 'Openrate', 'Clickrate', 'Optoutrate']
        display_df = agg_data[display_cols].copy()
        
        # Format rates as percentages
        for col in ['Openrate', 'Clickrate', 'Optoutrate']:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}")
            
        st.dataframe(display_df, use_container_width=True)

# --- Batch Prediction UI ---
def create_batch_prediction_ui(model, model_version, include_preheader=True):
    """
    Create UI for batch prediction
    
    Parameters:
    -----------
    model : object
        Trained model with predict method
    model_version : str
        Model version string
    include_preheader : bool
        Whether model includes preheader features
    """
    st.subheader("Batch Prediction")
    st.info(f"Using model version {model_version}")
    
    # Tabs for input methods
    tab1, tab2 = st.tabs(["Upload CSV", "Manual Entry"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file:
            try:
                # Import data
                input_df = import_subjects_from_csv(uploaded_file)
                
                # Display preview
                st.subheader("Preview")
                st.dataframe(input_df.head(5), use_container_width=True)
                
                # Column mapping
                st.subheader("Column Mapping")
                
                # Subject column (required)
                subject_col = st.selectbox(
                    "Subject column",
                    options=input_df.columns,
                    index=input_df.columns.get_loc("Subject") if "Subject" in input_df.columns else 0
                )
                
                # Preheader column (optional)
                preheader_col = None
                if include_preheader:
                    preheader_options = ["None"] + list(input_df.columns)
                    selected_preheader = st.selectbox(
                        "Preheader column",
                        options=preheader_options,
                        index=preheader_options.index("Preheader") if "Preheader" in preheader_options else 0
                    )
                    if selected_preheader != "None":
                        preheader_col = selected_preheader
                
                # Campaign settings
                st.subheader("Campaign Settings")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Dialog options
                    dialog_options = sorted(DIALOG_VALUES.items())
                    selected_dialog_display = st.selectbox(
                        'Dialog',
                        options=[label for _, label in dialog_options],
                        index=0
                    )
                    selected_dialog_code = next(code[0] for key, (code, label) in dialog_options if label == selected_dialog_display)
                    
                    # Syfte options
                    syfte_options = sorted(SYFTE_VALUES.items())
                    selected_syfte_display = st.selectbox(
                        'Syfte',
                        options=[label for _, label in syfte_options],
                        index=0
                    )
                    selected_syfte_code = next(code[0] for key, (code, label) in syfte_options if label == selected_syfte_display)
                
                with col2:
                    # Product options
                    product_options = sorted(PRODUKT_VALUES.items())
                    selected_product_display = st.selectbox(
                        'Product',
                        options=[label for _, label in product_options],
                        index=0
                    )
                    selected_product_code = next(code[0] for key, (code, label) in product_options if label == selected_product_display)
                    
                    # Bolag options
                    bolag_options = sorted(DIALOG_VALUES.keys())
                    excluded_bolag_display = st.multiselect(
                        'Exclude Bolag',
                        options=bolag_options
                    )
                    included_bolag = [key for key in bolag_options if key not in excluded_bolag_display]
                
                col1, col2 = st.columns(2)
                with col1:
                    min_age = st.number_input('Min Age', min_value=18, max_value=100, value=18)
                with col2:
                    max_age = st.number_input('Max Age', min_value=18, max_value=100, value=100)
                
                # Run prediction
                if st.button("Run Batch Prediction"):
                    with st.spinner("Running predictions..."):
                        # Extract subject and preheader
                        subjects = input_df[subject_col].tolist()
                        preheaders = input_df[preheader_col].tolist() if preheader_col else None
                        
                        # Run prediction
                        results = batch_predict(
                            model=model,
                            subjects=subjects,
                            preheaders=preheaders,
                            dialog=selected_dialog_code,
                            syfte=selected_syfte_code,
                            product=selected_product_code,
                            min_age=min_age,
                            max_age=max_age,
                            bolag=included_bolag,
                            include_preheader=include_preheader
                        )
                        
                        # Display results
                        st.subheader("Prediction Results")
                        
                        # Format for display
                        display_df = results.copy()
                        display_df['Predicted_Openrate'] = display_df['Predicted_Openrate'].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Summary statistics
                        st.subheader("Summary")
                        
                        # Calculate statistics on numeric values
                        numeric_results = pd.to_numeric(results['Predicted_Openrate'], errors='coerce')
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Average Open Rate", f"{numeric_results.mean():.2%}")
                        col2.metric("Minimum Open Rate", f"{numeric_results.min():.2%}")
                        col3.metric("Maximum Open Rate", f"{numeric_results.max():.2%}")
                        
                        # Export button
                        csv_content = export_results_to_csv(results)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv_content,
                            file_name="prediction_results.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        st.write("Enter multiple subject lines for batch prediction")
        
        # Text area for multiple subjects
        subjects_text = st.text_area(
            "Enter subject lines (one per line)",
            height=150
        )
        
        # Preheader text area if model supports it
        preheaders_text = ""
        if include_preheader:
            preheaders_text = st.text_area(
                "Enter preheaders (one per line, must match number of subject lines)",
                height=150
            )
        
        # Campaign settings (same as in Upload CSV tab)
        st.subheader("Campaign Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dialog options
            dialog_options = sorted(DIALOG_VALUES.items())
            selected_dialog_display = st.selectbox(
                'Dialog',
                options=[label for _, label in dialog_options],
                index=0,
                key="manual_dialog"
            )
            selected_dialog_code = next(code[0] for key, (code, label) in dialog_options if label == selected_dialog_display)
            
            # Syfte options
            syfte_options = sorted(SYFTE_VALUES.items())
            selected_syfte_display = st.selectbox(
                'Syfte',
                options=[label for _, label in syfte_options],
                index=0,
                key="manual_syfte"
            )
            selected_syfte_code = next(code[0] for key, (code, label) in syfte_options if label == selected_syfte_display)
        
        with col2:
            # Product options
            product_options = sorted(PRODUKT_VALUES.items())
            selected_product_display = st.selectbox(
                'Product',
                options=[label for _, label in product_options],
                index=0,
                key="manual_product"
            )
            selected_product_code = next(code[0] for key, (code, label) in product_options if label == selected_product_display)
            
            # Bolag options
            bolag_options = sorted(DIALOG_VALUES.keys())
            excluded_bolag_display = st.multiselect(
                'Exclude Bolag',
                options=bolag_options,
                key="manual_bolag"
            )
            included_bolag = [key for key in bolag_options if key not in excluded_bolag_display]
        
        col1, col2 = st.columns(2)
        with col1:
            min_age = st.number_input('Min Age', min_value=18, max_value=100, value=18, key="manual_min_age")
        with col2:
            max_age = st.number_input('Max Age', min_value=18, max_value=100, value=100, key="manual_max_age")
        
        # Run prediction
        if st.button("Run Batch Prediction", key="manual_predict_button"):
            # Check if there's input
            if not subjects_text.strip():
                st.error("Please enter at least one subject line")
                return
            
            # Parse input
            subjects = [line.strip() for line in subjects_text.split('\n') if line.strip()]
            
            if include_preheader:
                preheaders = [line.strip() for line in preheaders_text.split('\n') if line.strip()]
                
                # Check if preheaders match subjects
                if len(preheaders) > 0 and len(preheaders) != len(subjects):
                    st.error(f"Number of preheaders ({len(preheaders)}) must match number of subjects ({len(subjects)})")
                    return
                
                # If no preheaders provided, use empty strings
                if len(preheaders) == 0:
                    preheaders = [''] * len(subjects)
            else:
                preheaders = None
            
            with st.spinner("Running predictions..."):
                # Run prediction
                results = batch_predict(
                    model=model,
                    subjects=subjects,
                    preheaders=preheaders,
                    dialog=selected_dialog_code,
                    syfte=selected_syfte_code,
                    product=selected_product_code,
                    min_age=min_age,
                    max_age=max_age,
                    bolag=included_bolag,
                    include_preheader=include_preheader
                )
                
                # Display results
                st.subheader("Prediction Results")
                
                # Format for display
                display_df = results.copy()
                display_df['Predicted_Openrate'] = display_df['Predicted_Openrate'].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary")
                
                # Calculate statistics on numeric values
                numeric_results = pd.to_numeric(results['Predicted_Openrate'], errors='coerce')
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Open Rate", f"{numeric_results.mean():.2%}")
                col2.metric("Minimum Open Rate", f"{numeric_results.min():.2%}")
                col3.metric("Maximum Open Rate", f"{numeric_results.max():.2%}")
                
                # Export button
                csv_content = export_results_to_csv(results)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_content,
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )
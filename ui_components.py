"""
Enhanced UI components for multi-KPI predictor application
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Tuple, Optional, Union

from utils import (
    logger, DIALOG_VALUES, SYFTE_VALUES, PRODUKT_VALUES, categorize_age,
    import_subjects_from_csv, export_results_to_csv
)
from enhanced_features import create_campaign_level_features


def create_enhanced_input_ui():
    """Create an enhanced UI for subject line and preheader input with real-time feedback"""
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
            use_genai = st.checkbox('Generate alternatives with AI', value=True)
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


def display_multi_kpi_results(options, best_option, improvement):
    """
    Display results of multi-KPI optimization
    
    Parameters:
    -----------
    options : List[Dict[str, Any]]
        List of options with predictions
    best_option : Dict[str, Any]
        Best option selected
    improvement : Dict[str, float]
        Improvement metrics
    """
    st.subheader("Subject Line & Preheader Optimization")
    
    # Display best option
    st.success(f"🏆 Best Option (Version {best_option['version']})")
    
    # Create a card for the best option
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Subject Line:**")
            st.markdown(f"```\n{best_option['subject']}\n```")
            
            st.markdown("**Preheader:**")
            st.markdown(f"```\n{best_option['preheader']}\n```")
        
        with col2:
            # Display improvement metrics
            if best_option['version'] != 'A':  # Only show improvement if not baseline
                openrate_imp = improvement.get('openrate', 0)
                clickrate_imp = improvement.get('clickrate', 0)
                optoutrate_imp = improvement.get('optoutrate', 0)
                
                st.metric("Open Rate Improvement", f"{openrate_imp:.2%}", delta=f"{openrate_imp:.2%}")
                st.metric("Click Rate Improvement", f"{clickrate_imp:.2%}", delta=f"{clickrate_imp:.2%}")
                st.metric("Opt-out Rate Improvement", f"{-optoutrate_imp:.2%}", delta=f"{-optoutrate_imp:.2%}")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["All Options", "Comparison"])
    
    with tab1:
        # Display all options in a table
        for i, option in enumerate(options):
            version = option['version']
            is_best = version == best_option['version']
            
            with st.expander(f"Version {version}" + (" (Best)" if is_best else ""), expanded=is_best):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("**Subject Line:**")
                    st.code(option['subject'], language=None)
                    
                    st.markdown("**Preheader:**")
                    st.code(option['preheader'], language=None)
                
                with col2:
                    # Display metrics
                    openrate = option['predictions'].get('openrate', 0)
                    clickrate = option['predictions'].get('clickrate', 0)
                    optoutrate = option['predictions'].get('optoutrate', 0)
                    
                    st.metric("Open Rate", f"{openrate:.2%}")
                    st.metric("Click Rate", f"{clickrate:.2%}")
                    st.metric("Opt-out Rate", f"{optoutrate:.2%}")
                    
                    # Show score
                    st.metric("Combined Score", f"{option.get('combined_score', 0):.2f}")
    
    with tab2:
        # Create comparison visualizations
        
        # Extract data for charts
        versions = [o['version'] for o in options]
        openrates = [o['predictions'].get('openrate', 0) for o in options]
        clickrates = [o['predictions'].get('clickrate', 0) for o in options]
        optoutrates = [o['predictions'].get('optoutrate', 0) for o in options]
        
        # Bar chart for open rates
        fig1 = go.Figure()
        colors = ['#a2b9bc' if v != best_option['version'] else '#2f93e0' for v in versions]
        
        fig1.add_trace(go.Bar(
            x=versions,
            y=openrates,
            text=[f"{r:.2%}" for r in openrates],
            textposition='auto',
            marker_color=colors,
            name='Open Rate'
        ))
        
        fig1.update_layout(
            title="Open Rate Comparison",
            xaxis_title="Version",
            yaxis_title="Open Rate",
            yaxis_tickformat=".0%",
            height=400
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Side-by-side chart for click and opt-out
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            x=versions,
            y=clickrates,
            text=[f"{r:.2%}" for r in clickrates],
            textposition='auto',
            marker_color='#32a852',
            name='Click Rate'
        ))
        
        fig2.add_trace(go.Bar(
            x=versions,
            y=optoutrates,
            text=[f"{r:.2%}" for r in optoutrates],
            textposition='auto',
            marker_color='#e74c3c',
            name='Opt-out Rate'
        ))
        
        fig2.update_layout(
            title="Click & Opt-out Rate Comparison",
            xaxis_title="Version",
            yaxis_title="Rate",
            yaxis_tickformat=".0%",
            height=400,
            barmode='group'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Create a comparison table
        comparison_df = pd.DataFrame({
            'Version': versions,
            'Subject Line': [o['subject'] for o in options],
            'Open Rate': [f"{o['predictions'].get('openrate', 0):.2%}" for o in options],
            'Click Rate': [f"{o['predictions'].get('clickrate', 0):.2%}" for o in options],
            'Opt-out Rate': [f"{o['predictions'].get('optoutrate', 0):.2%}" for o in options],
            'Combined Score': [f"{o.get('combined_score', 0):.2f}" for o in options]
        })
        
        st.dataframe(comparison_df, use_container_width=True)


def create_kpi_dashboard(delivery_data):
    """
    Create an interactive KPI dashboard with multiple views and insights
    
    Parameters:
    -----------
    delivery_data : DataFrame
        Delivery data with campaign information
    """
    st.header("Email Campaign KPI Dashboard")
    
    # Calculate key metrics
    avg_open_rate = delivery_data['Openrate'].mean()
    avg_click_rate = delivery_data['Clickrate'].mean()
    avg_optout_rate = delivery_data['Optoutrate'].mean()
    
    total_sendouts = delivery_data['Sendouts'].sum()
    total_opens = delivery_data['Opens'].sum()
    total_clicks = delivery_data['Clicks'].sum()
    total_optouts = delivery_data['Optouts'].sum()
    
    # Top row metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Open Rate", f"{avg_open_rate:.2%}")
        st.caption(f"Total Opens: {total_opens:,}")
    
    with col2:
        st.metric("Avg Click Rate", f"{avg_click_rate:.2%}")
        st.caption(f"Total Clicks: {total_clicks:,}")
    
    with col3:
        st.metric("Avg Opt-out Rate", f"{avg_optout_rate:.2%}")
        st.caption(f"Total Opt-outs: {total_optouts:,}")
    
    with col4:
        st.metric("Total Campaigns", f"{len(delivery_data):,}")
        st.caption(f"Total Sendouts: {total_sendouts:,}")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["KPI Trends", "Category Analysis", "Campaign Details"])
    
    with tab1:
        # KPI Trends view
        st.subheader("KPI Distribution")
        
        # Select KPI to visualize
        kpi_options = {
            "Openrate": "Open Rate", 
            "Clickrate": "Click Rate", 
            "Optoutrate": "Opt-out Rate"
        }
        
        selected_kpi = st.radio(
            "Select KPI to analyze",
            options=list(kpi_options.keys()),
            format_func=lambda x: kpi_options[x],
            horizontal=True
        )
        
        # Distribution histogram
        fig = px.histogram(
            delivery_data,
            x=selected_kpi,
            nbins=20,
            histnorm='percent',
            labels={selected_kpi: kpi_options[selected_kpi]},
            title=f"{kpi_options[selected_kpi]} Distribution"
        )
        
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Time trend if date is available
        if 'Date' in delivery_data.columns:
            st.subheader("KPI Trends Over Time")
            
            # Ensure date is datetime
            if not pd.api.types.is_datetime64_any_dtype(delivery_data['Date']):
                delivery_data['Date'] = pd.to_datetime(delivery_data['Date'], errors='coerce')
            
            # Group by month
            delivery_data['Month'] = delivery_data['Date'].dt.to_period('M')
            monthly_kpis = delivery_data.groupby('Month').agg({
                'Openrate': 'mean',
                'Clickrate': 'mean',
                'Optoutrate': 'mean',
                'Sendouts': 'sum'
            }).reset_index()
            
            monthly_kpis['Month'] = monthly_kpis['Month'].astype(str)
            
            # Create trend line chart
            fig = px.line(
                monthly_kpis,
                x='Month',
                y=[selected_kpi],
                labels={selected_kpi: kpi_options[selected_kpi], 'Month': 'Month'},
                title=f"{kpi_options[selected_kpi]} Trend by Month"
            )
            
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Category Analysis view
        st.subheader("Performance by Category")
        
        # Select category to analyze
        category_options = {"Dialog": "Dialog", "Syfte": "Syfte", "Product": "Product"}
        selected_category = st.selectbox(
            "Select category to analyze",
            options=list(category_options.keys()),
            format_func=lambda x: category_options[x]
        )
        
        # Group by selected category
        category_kpis = delivery_data.groupby(selected_category).agg({
            'Openrate': 'mean',
            'Clickrate': 'mean',
            'Optoutrate': 'mean',
            'Sendouts': 'sum',
            'InternalName': 'count'
        }).reset_index()
        
        category_kpis = category_kpis.rename(columns={'InternalName': 'CampaignCount'})
        category_kpis = category_kpis.sort_values('Openrate', ascending=False)
        
        # Display metrics by category
        st.dataframe(
            category_kpis.style.format({
                'Openrate': '{:.2%}',
                'Clickrate': '{:.2%}',
                'Optoutrate': '{:.2%}',
                'Sendouts': '{:,.0f}',
                'CampaignCount': '{:,.0f}'
            }),
            use_container_width=True
        )
        
        # Bar chart comparison
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=category_kpis[selected_category],
            x=category_kpis['Openrate'],
            text=[f"{r:.2%}" for r in category_kpis['Openrate']],
            textposition='auto',
            marker_color='#1f77b4',
            name='Open Rate',
            orientation='h'
        ))
        
        fig.update_layout(
            title=f"Open Rate by {selected_category}",
            xaxis_title="Open Rate",
            yaxis_title=selected_category,
            xaxis_tickformat=".0%",
            height=max(400, len(category_kpis) * 25)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Multi-KPI comparison
        kpi_fig = px.bar(
            category_kpis,
            y=selected_category,
            x=['Openrate', 'Clickrate', 'Optoutrate'],
            labels={
                'value': 'Rate',
                'variable': 'KPI'
            },
            title=f"All KPIs by {selected_category}",
            orientation='h',
            barmode='group',
            height=max(500, len(category_kpis) * 30)
        )
        
        kpi_fig.update_xaxes(tickformat=".0%")
        
        st.plotly_chart(kpi_fig, use_container_width=True)
    
    with tab3:
        # Campaign Details view
        st.subheader("Top Performing Campaigns")
        
        # Metrics options
        metric_options = {
            "Openrate": "Open Rate", 
            "Clickrate": "Click Rate", 
            "Sendouts": "Sendout Volume",
            "combined": "Combined Score (Open + Click - Optout)"
        }
        
        selected_metric = st.selectbox(
            "Rank by",
            options=list(metric_options.keys()),
            format_func=lambda x: metric_options[x]
        )
        
        # Calculate combined score if selected
        if selected_metric == 'combined':
            delivery_data['CombinedScore'] = delivery_data['Openrate'] + delivery_data['Clickrate'] - delivery_data['Optoutrate']
            sort_col = 'CombinedScore'
        else:
            sort_col = selected_metric
        
        # Get top campaigns
        top_n = st.slider("Number of campaigns to show", 5, 20, 10)
        
        top_campaigns = delivery_data.sort_values(sort_col, ascending=False).head(top_n)[
            ['InternalName', 'Subject', 'Preheader', 'Openrate', 'Clickrate', 'Optoutrate', 'Sendouts', 'Dialog', 'Syfte', 'Product']
        ]
        
        # Display top campaigns
        st.dataframe(
            top_campaigns.style.format({
                'Openrate': '{:.2%}',
                'Clickrate': '{:.2%}',
                'Optoutrate': '{:.2%}',
                'Sendouts': '{:,.0f}'
            }),
            use_container_width=True
        )
        
        # Subject line analysis
        st.subheader("Subject Line Analysis")
        
        # Text features to analyze
        text_features = {
            "Length": lambda s: len(s),
            "Word Count": lambda s: len(s.split()),
            "Has Question Mark": lambda s: '?' in s,
            "Has Exclamation": lambda s: '!' in s,
            "Contains Number": lambda s: any(c.isdigit() for c in s)
        }
        
        selected_feature = st.selectbox(
            "Analyze subject lines by",
            options=list(text_features.keys())
        )
        
        # Apply the feature function
        feature_func = text_features[selected_feature]
        
        if selected_feature in ["Has Question Mark", "Has Exclamation", "Contains Number"]:
            # Binary features
            delivery_data['FeatureValue'] = delivery_data['Subject'].apply(feature_func)
            
            # Group by feature
            feature_kpis = delivery_data.groupby('FeatureValue').agg({
                'Openrate': 'mean',
                'Clickrate': 'mean',
                'Optoutrate': 'mean',
                'InternalName': 'count'
            }).reset_index()
            
            feature_kpis['FeatureValue'] = feature_kpis['FeatureValue'].map({True: "Yes", False: "No"})
            feature_kpis = feature_kpis.rename(columns={'InternalName': 'CampaignCount'})
            
            # Bar chart
            fig = px.bar(
                feature_kpis,
                x='FeatureValue',
                y=['Openrate', 'Clickrate', 'Optoutrate'],
                barmode='group',
                title=f"KPIs by Subject Line {selected_feature}",
                labels={'FeatureValue': selected_feature, 'value': 'Rate', 'variable': 'KPI'}
            )
            
            fig.update_yaxes(tickformat=".0%")
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Numeric features
            delivery_data['FeatureValue'] = delivery_data['Subject'].apply(feature_func)
            
            # Scatter plot
            fig = px.scatter(
                delivery_data,
                x='FeatureValue',
                y='Openrate',
                title=f"Open Rate by Subject {selected_feature}",
                labels={'FeatureValue': selected_feature, 'Openrate': 'Open Rate'},
                trendline="ols"
            )
            
            fig.update_yaxes(tickformat=".0%")
            
            st.plotly_chart(fig, use_container_width=True)


def create_batch_prediction_ui(model_manager, version=None):
    """
    Create UI for batch prediction of multiple email subject lines and preheaders
    
    Parameters:
    -----------
    model_manager : MultiKpiModelManager
        Model manager with trained models
    version : str, optional
        Model version string
    """
    st.header("Batch KPI Prediction")
    st.info(f"Using model version {version}")
    
    # Create tabs for input methods
    tab1, tab2 = st.tabs(["Upload CSV", "Manual Entry"])
    
    with tab1:
        st.subheader("Upload Subject Lines")
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
                preheader_options = ["None"] + list(input_df.columns)
                selected_preheader = st.selectbox(
                    "Preheader column",
                    options=preheader_options,
                    index=preheader_options.index("Preheader") if "Preheader" in preheader_options else 0
                )
                
                preheader_col = None if selected_preheader == "None" else selected_preheader
                
                # Campaign settings
                st.subheader("Campaign Settings")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Dialog options
                    dialog_options = sorted(DIALOG_VALUES.items())
                    dialog_labels = [(code[0], label) for key, (code, label) in dialog_options]
                    
                    selected_dialog_display = st.selectbox(
                        'Dialog',
                        options=[label for _, label in dialog_labels],
                        key="batch_dialog"
                    )
                    selected_dialog_code = next(code for code, label in dialog_labels if label == selected_dialog_display)
                    
                    # Syfte options
                    syfte_options = sorted(SYFTE_VALUES.items())
                    syfte_labels = [(code[0], label) for key, (code, label) in syfte_options]
                    
                    selected_syfte_display = st.selectbox(
                        'Syfte',
                        options=[label for _, label in syfte_labels],
                        key="batch_syfte"
                    )
                    selected_syfte_code = next(code for code, label in syfte_labels if label == selected_syfte_display)
                
                with col2:
                    # Product options
                    product_options = sorted(PRODUKT_VALUES.items())
                    product_labels = [(code[0], label) for key, (code, label) in product_options]
                    
                    selected_product_display = st.selectbox(
                        'Product',
                        options=[label for _, label in product_labels],
                        key="batch_product"
                    )
                    selected_product_code = next(code for code, label in product_labels if label == selected_product_display)
                    
                    # Bolag options
                    bolag_options = sorted(DIALOG_VALUES.keys())
                    excluded_bolag_display = st.multiselect(
                        'Exclude Bolag',
                        options=bolag_options,
                        key="batch_bolag"
                    )
                    included_bolag = [key for key in bolag_options if key not in excluded_bolag_display]
                
                col1, col2 = st.columns(2)
                with col1:
                    min_age = st.number_input('Min Age', min_value=18, max_value=100, value=18, key="batch_min_age")
                with col2:
                    max_age = st.number_input('Max Age', min_value=18, max_value=100, value=100, key="batch_max_age")
                
                # Campaign metadata
                campaign_metadata = {
                    'dialog': selected_dialog_code,
                    'dialog_display': selected_dialog_display,
                    'syfte': selected_syfte_code,
                    'syfte_display': selected_syfte_display,
                    'product': selected_product_code,
                    'product_display': selected_product_display,
                    'min_age': min_age,
                    'max_age': max_age,
                    'included_bolag': included_bolag
                }
                
                # Run prediction
                if st.button("Run Batch Prediction", key="batch_csv_button"):
                    with st.spinner("Running predictions..."):
                        # Get list of subjects and preheaders
                        subjects = input_df[subject_col].tolist()
                        preheaders = input_df[preheader_col].tolist() if preheader_col else ['' for _ in subjects]
                        
                        # Run predictions
                        results = []
                        
                        for i, (subject, preheader) in enumerate(zip(subjects, preheaders)):
                            # Make predictions
                            predictions = model_manager.predict_new_campaign(
                                subject, preheader, campaign_metadata, version
                            )
                            
                            # Calculate combined score
                            openrate = predictions.get('openrate', 0)
                            clickrate = predictions.get('clickrate', 0)
                            optoutrate = predictions.get('optoutrate', 0)
                            combined_score = 0.5 * openrate + 0.5 * clickrate - 2.0 * optoutrate
                            
                            # Add to results
                            results.append({
                                'Subject': subject,
                                'Preheader': preheader,
                                'Predicted_Openrate': openrate,
                                'Predicted_Clickrate': clickrate,
                                'Predicted_Optoutrate': optoutrate,
                                'Combined_Score': combined_score
                            })
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.subheader("Prediction Results")
                        
                        # Sort by combined score
                        results_df = results_df.sort_values('Combined_Score', ascending=False)
                        
                        # Format for display
                        display_df = results_df.copy()
                        for col in ['Predicted_Openrate', 'Predicted_Clickrate', 'Predicted_Optoutrate']:
                            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Summary statistics
                        st.subheader("Summary")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Avg Open Rate", f"{results_df['Predicted_Openrate'].mean():.2%}")
                        col2.metric("Avg Click Rate", f"{results_df['Predicted_Clickrate'].mean():.2%}")
                        col3.metric("Avg Opt-out Rate", f"{results_df['Predicted_Optoutrate'].mean():.2%}")
                        
                        # Visualization
                        st.subheader("Visualization")
                        
                        # Distribution of open rates
                        fig = px.histogram(
                            results_df,
                            x='Predicted_Openrate',
                            nbins=20,
                            labels={'Predicted_Openrate': 'Predicted Open Rate'},
                            title="Distribution of Predicted Open Rates"
                        )
                        
                        fig.update_xaxes(tickformat=".0%")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Export button
                        csv_content = export_results_to_csv(results_df)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv_content,
                            file_name="batch_prediction_results.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                logger.error(f"Error processing file: {str(e)}")
                logger.error(traceback.format_exc())
    
    with tab2:
        st.subheader("Enter Subject Lines")
        
        # Text area for multiple subjects
        subjects_text = st.text_area(
            "Enter subject lines (one per line)",
            height=150
        )
        
        # Preheader text area
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
            dialog_labels = [(code[0], label) for key, (code, label) in dialog_options]
            
            selected_dialog_display = st.selectbox(
                'Dialog',
                options=[label for _, label in dialog_labels],
                key="manual_dialog"
            )
            selected_dialog_code = next(code for code, label in dialog_labels if label == selected_dialog_display)
            
            # Syfte options
            syfte_options = sorted(SYFTE_VALUES.items())
            syfte_labels = [(code[0], label) for key, (code, label) in syfte_options]
            
            selected_syfte_display = st.selectbox(
                'Syfte',
                options=[label for _, label in syfte_labels],
                key="manual_syfte"
            )
            selected_syfte_code = next(code for code, label in syfte_labels if label == selected_syfte_display)
        
        with col2:
            # Product options
            product_options = sorted(PRODUKT_VALUES.items())
            product_labels = [(code[0], label) for key, (code, label) in product_options]
            
            selected_product_display = st.selectbox(
                'Product',
                options=[label for _, label in product_labels],
                key="manual_product"
            )
            selected_product_code = next(code for code, label in product_labels if label == selected_product_display)
            
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
        
        # Campaign metadata
        campaign_metadata = {
            'dialog': selected_dialog_code,
            'dialog_display': selected_dialog_display,
            'syfte': selected_syfte_code,
            'syfte_display': selected_syfte_display,
            'product': selected_product_code,
            'product_display': selected_product_display,
            'min_age': min_age,
            'max_age': max_age,
            'included_bolag': included_bolag
        }
        
        # Run prediction
        if st.button("Run Batch Prediction", key="manual_predict_button"):
            # Check if there's input
            if not subjects_text.strip():
                st.error("Please enter at least one subject line")
                return
            
            # Parse input
            subjects = [line.strip() for line in subjects_text.split('\n') if line.strip()]
            
            preheaders = [line.strip() for line in preheaders_text.split('\n') if line.strip()]
            
            # Check if preheaders match subjects
            if preheaders and len(preheaders) != len(subjects):
                st.error(f"Number of preheaders ({len(preheaders)}) must match number of subjects ({len(subjects)})")
                return
            
            # If no preheaders provided, use empty strings
            if not preheaders:
                preheaders = [''] * len(subjects)
            
            with st.spinner("Running predictions..."):
                # Run predictions
                results = []
                
                for i, (subject, preheader) in enumerate(zip(subjects, preheaders)):
                    # Make predictions
                    predictions = model_manager.predict_new_campaign(
                        subject, preheader, campaign_metadata, version
                    )
                    
                    # Calculate combined score
                    openrate = predictions.get('openrate', 0)
                    clickrate = predictions.get('clickrate', 0)
                    optoutrate = predictions.get('optoutrate', 0)
                    combined_score = 0.5 * openrate + 0.5 * clickrate - 2.0 * optoutrate
                    
                    # Add to results
                    results.append({
                        'Subject': subject,
                        'Preheader': preheader,
                        'Predicted_Openrate': openrate,
                        'Predicted_Clickrate': clickrate,
                        'Predicted_Optoutrate': optoutrate,
                        'Combined_Score': combined_score
                    })
                
                # Create results DataFrame
                results_df = pd.DataFrame(results)
                
                # Display results
                st.subheader("Prediction Results")
                
                # Sort by combined score
                results_df = results_df.sort_values('Combined_Score', ascending=False)
                
                # Format for display
                display_df = results_df.copy()
                for col in ['Predicted_Openrate', 'Predicted_Clickrate', 'Predicted_Optoutrate']:
                    display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Open Rate", f"{results_df['Predicted_Openrate'].mean():.2%}")
                col2.metric("Avg Click Rate", f"{results_df['Predicted_Clickrate'].mean():.2%}")
                col3.metric("Avg Opt-out Rate", f"{results_df['Predicted_Optoutrate'].mean():.2%}")
                
                # Export button
                csv_content = export_results_to_csv(results_df)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_content,
                    file_name="batch_prediction_results.csv",
                    mime="text/csv"
                )


def create_model_comparison_ui(model_manager, delivery_data, customer_data, version=None):
    """
    Create UI for model comparison and performance analysis
    
    Parameters:
    -----------
    model_manager : MultiKpiModelManager
        Model manager with trained models
    delivery_data : DataFrame
        Delivery data with campaign information
    customer_data : DataFrame
        Customer data with demographic information
    version : str, optional
        Model version string
    """
    st.header("Model Performance")
    
    # Get best model info
    best_model_info = model_manager.get_best_model_info()
    
    if not best_model_info:
        st.warning("No model performance data available.")
        return
    
    # Create tabs for different KPI models
    kpi_types = list(best_model_info.keys())
    
    if not kpi_types:
        st.warning("No KPI models found.")
        return
    
    tabs = st.tabs([kpi.capitalize() for kpi in kpi_types])
    
    for i, kpi_type in enumerate(kpi_types):
        with tabs[i]:
            info = best_model_info[kpi_type]
            
            st.subheader(f"{kpi_type.capitalize()} Model Performance")
            
            # Model details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Model Type:** {info.get('model_type', 'Unknown').upper()}")
                st.markdown(f"**Version:** {version}")
            
            with col2:
                performance = info.get('performance', {})
                r2 = performance.get('r2', 0)
                rmse = performance.get('rmse', 0)
                
                st.metric("R² Score", f"{r2:.4f}")
                st.metric("RMSE", f"{rmse:.6f}")
            
            # Feature importance if available
            model = model_manager._get_model(kpi_type, version)
            
            if model and hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_'):
                st.subheader("Feature Importance")
                
                # Get feature importances
                importances = model.feature_importances_
                feature_names = model.feature_names_
                
                # Create DataFrame for plotting
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Plot top 15 features
                top_n = min(15, len(importance_df))
                top_features = importance_df.head(top_n)
                
                fig = px.bar(
                    top_features,
                    y='Feature',
                    x='Importance',
                    orientation='h',
                    title=f"Top {top_n} Features for {kpi_type.capitalize()} Model"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show feature importance table
                with st.expander("View all feature importances"):
                    st.dataframe(importance_df, use_container_width=True)
            
            # Scatter plot of predictions vs actual if available
            # (This would require re-running predictions on the test set)
            st.subheader("Recent Model Improvements")
            
            # Placeholder for model improvement history
            improvement_data = {
                'Version': ['24.01.01', '24.02.15', version],
                'R2 Score': [0.42, 0.48, r2],
                'RMSE': [0.12, 0.09, rmse],
                'Model Type': ['XGBoost', 'LightGBM', info.get('model_type', 'Unknown').upper()]
            }
            
            improvement_df = pd.DataFrame(improvement_data)
            
            # Plot improvement over versions
            fig = px.line(
                improvement_df,
                x='Version',
                y='R2 Score',
                title="Model R² Score Improvement Over Versions",
                markers=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display improvement table
            st.dataframe(improvement_df, use_container_width=True)
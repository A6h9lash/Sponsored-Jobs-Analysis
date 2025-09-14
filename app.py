#!/usr/bin/env python3
"""
Sponsored Jobs Analysis Engine
A modern, intuitive web-based analysis tool for H1B visa sponsorship data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import html
from typing import Dict, List, Tuple
import warnings
import requests
import io
from urllib.parse import urlparse
import psycopg2
from sqlalchemy import create_engine
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sponsored Jobs Analysis Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Reusable pagination function
def paginate_dataframe(df, page_size=25, page_key="page"):
    """Add pagination controls to a dataframe"""
    if df.empty:
        return df, 0, 0
    
    # Get current page from session state
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    
    current_page = st.session_state[page_key]
    total_rows = len(df)
    total_pages = (total_rows - 1) // page_size + 1
    
    # Ensure current page is valid
    if current_page >= total_pages:
        current_page = 0
        st.session_state[page_key] = 0
    
    # Calculate start and end indices
    start_idx = current_page * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    # Get the page data
    page_data = df.iloc[start_idx:end_idx]
    
    return page_data, current_page, total_pages

# Reusable pagination controls function
def display_pagination_controls(current_page, total_pages, page_key="page"):
    """Display pagination controls"""
    if total_pages <= 1:
        return
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⏮️ First", key=f"{page_key}_first"):
            st.session_state[page_key] = 0
            st.rerun()
    
    with col2:
        if st.button("⬅️ Previous", key=f"{page_key}_prev"):
            if st.session_state[page_key] > 0:
                st.session_state[page_key] -= 1
                st.rerun()
    
    with col3:
        st.write(f"**Page {current_page + 1} of {total_pages}**")
    
    with col4:
        if st.button("Next ➡️", key=f"{page_key}_next"):
            if st.session_state[page_key] < total_pages - 1:
                st.session_state[page_key] += 1
                st.rerun()
    
    with col5:
        if st.button("Last ⏭️", key=f"{page_key}_last"):
            st.session_state[page_key] = total_pages - 1
            st.rerun()

# Reusable clickable table function
def create_clickable_table(df, max_rows=1000):
    """Create a table with clickable links using Streamlit's native components"""
    try:
        # Limit rows for performance
        df_display = df.head(max_rows)
        
        # Create a copy for display
        display_df = df_display.copy()
        
        # Keep URLs as-is since we're using Streamlit's LinkColumn
        # The LinkColumn will handle the display and clicking
        
        return display_df
        
    except Exception as e:
        # Fallback to simple dataframe display
        st.error(f"Error creating table: {str(e)}")
        return df.head(100)  # Return first 100 rows as fallback

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sponsored-high {
        color: #28a745;
        font-weight: bold;
    }
    .sponsored-low {
        color: #dc3545;
        font-weight: bold;
    }
    .sponsored-moderate {
        color: #ffc107;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Enhanced table scrolling */
    .stDataFrame {
        overflow-x: auto !important;
        overflow-y: auto !important;
    }
    
    .stDataFrame > div {
        overflow-x: auto !important;
        overflow-y: auto !important;
    }
    
    .stDataFrame table {
        min-width: 100% !important;
    }
    
    /* Custom scrollbar styling */
    .stDataFrame::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    .stDataFrame::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    .stDataFrame::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    .stDataFrame::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Custom table styling for clickable links */
    .custom-table {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 14px;
        line-height: 1.4;
    }
    
    .custom-table th {
        background-color: #f0f2f6 !important;
        color: #262730 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 12px 8px !important;
    }
    
    .custom-table td {
        color: #262730 !important;
        font-size: 14px !important;
        padding: 10px 8px !important;
        background-color: #ffffff !important;
        border-bottom: 1px solid #e6e9ef !important;
    }
    
    .custom-table tr:nth-child(even) td {
        background-color: #f8f9fa !important;
    }
    
    .custom-table tr:hover td {
        background-color: #e3f2fd !important;
    }
    
    .custom-table a {
        color: #1f77b4 !important;
        text-decoration: none !important;
        font-weight: 500 !important;
    }
    
    .custom-table a:hover {
        color: #0d5aa7 !important;
        text-decoration: underline !important;
    }
</style>
""", unsafe_allow_html=True)

# Configuration for data sources
DATA_SOURCES = {
    "google_drive": "https://drive.google.com/uc?export=download&id=YOUR_GOOGLE_DRIVE_FILE_ID_HERE",
    "postgresql": {
        "connection_string": "postgresql://postgres:5gsoJG24_qmCnSqM@34.132.93.79:5432/solwizz",
        "sql_query": """
        SELECT j.*, jr.id AS "jobRoleId", jr.name as "jobRoleName"
        FROM public."Job" j
        LEFT JOIN public."JobRole" jr ON j."roleId" = jr.id
        ORDER BY j.id ASC;
        """
    }
}

@st.cache_data
def load_parquet_data(file_path: str):
    """Load and cache parquet data from Google Drive URL"""
    try:
        st.info("📥 Loading data from Google Drive... This may take a moment for large files.")
        
        # For Google Drive links, we need to handle the download properly
        if 'drive.google.com' in file_path:
            # Extract file ID from Google Drive URL
            file_id = file_path.split('id=')[1].split('&')[0] if 'id=' in file_path else None
            if file_id:
                # Use direct download link for Google Drive
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            else:
                download_url = file_path
        else:
            download_url = file_path
        
        # Download the file
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Read the content into a BytesIO object
        content = io.BytesIO(response.content)
        
        # Load parquet from BytesIO
        return pd.read_parquet(content)
            
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download data from Google Drive: {e}")
        st.info("💡 **Troubleshooting Tips:**")
        st.info("1. Make sure the Google Drive link is set to 'Anyone with the link can view'")
        st.info("2. Use the direct download link format: https://drive.google.com/uc?export=download&id=YOUR_FILE_ID")
        st.info("3. Check your internet connection")
        raise
    except Exception as e:
        st.error(f"Error loading data from Google Drive: {e}")
        raise

@st.cache_data
def load_postgresql_data(db_config: dict):
    """Load and cache data from PostgreSQL database"""
    try:
        st.info("🗄️ Loading data from PostgreSQL database...")
        
        # Create SQLAlchemy engine using the connection string
        engine = create_engine(db_config['connection_string'])
        
        # Load data from database using the custom SQL query
        df = pd.read_sql_query(db_config['sql_query'], engine)
        
        # Close the engine
        engine.dispose()
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load data from PostgreSQL: {e}")
        st.info("💡 **Troubleshooting Tips:**")
        st.info("1. Check your database connection parameters")
        st.info("2. Ensure the tables 'Job' and 'JobRole' exist and have the correct structure")
        st.info("3. Verify your database credentials and permissions")
        st.info("4. Make sure your database server is accessible")
        st.info("5. Check if the JOIN between Job and JobRole tables is working correctly")
        raise

class SponsoredJobsAnalyzer:
    def __init__(self, data_source: str = "google_drive"):
        """Initialize the analyzer with data source configuration"""
        self.data_source = data_source
        self.data_config = DATA_SOURCES.get(data_source, DATA_SOURCES["google_drive"])
        self.df = None
        self.roles = None
        self.companies = None
        
    def load_data(self) -> bool:
        """Load and cache data from Google Drive or PostgreSQL"""
        try:
            # Load data based on source type
            if self.data_source == "google_drive":
                with st.spinner("Loading data from Google Drive... This may take a few minutes for large files."):
                    self.df = load_parquet_data(self.data_config)
            elif self.data_source == "postgresql":
                with st.spinner("Loading data from PostgreSQL database..."):
                    self.df = load_postgresql_data(self.data_config)
            else:
                st.error(f"Unknown data source: {self.data_source}")
                return False
            
            # Check required columns
            required_columns = ['Sponsored Job?', 'jobRoleName']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return False
                
            # Get unique values
            self.roles = sorted(self.df['jobRoleName'].dropna().unique().tolist())
            self.companies = sorted(self.df['company'].dropna().unique().tolist())
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def get_overall_insights(self) -> Dict:
        """Calculate overall visa sponsorship insights"""
        if self.df is None:
            return {}
            
        total_count = len(self.df)
        sponsored_count = (self.df['Sponsored Job?'] == 'Yes').sum()
        non_sponsored_count = total_count - sponsored_count
        sponsorship_rate = (sponsored_count / total_count) * 100 if total_count > 0 else 0
        
        return {
            'total_jobs': total_count,
            'sponsored_jobs': sponsored_count,
            'non_sponsored_jobs': non_sponsored_count,
            'sponsorship_rate': sponsorship_rate
        }
    
    def get_role_insights(self, role_name: str) -> Dict:
        """Calculate role-specific insights"""
        if self.df is None or role_name not in self.roles:
            return {}
            
        role_data = self.df[self.df['jobRoleName'] == role_name]
        
        if len(role_data) == 0:
            return {}
            
        total_count = len(role_data)
        sponsored_count = (role_data['Sponsored Job?'] == 'Yes').sum()
        non_sponsored_count = total_count - sponsored_count
        sponsorship_rate = (sponsored_count / total_count) * 100 if total_count > 0 else 0
        
        companies = role_data['company'].nunique()
        locations = role_data['location'].nunique()
        
        return {
            'role_name': role_name,
            'total_jobs': total_count,
            'sponsored_jobs': sponsored_count,
            'non_sponsored_jobs': non_sponsored_count,
            'sponsorship_rate': sponsorship_rate,
            'unique_companies': companies,
            'unique_locations': locations
        }
    
    def get_company_insights(self, company_name: str) -> Dict:
        """Calculate company-specific insights"""
        if self.df is None or company_name not in self.companies:
            return {}
            
        company_data = self.df[self.df['company'] == company_name]
        
        if len(company_data) == 0:
            return {}
            
        total_count = len(company_data)
        sponsored_count = (company_data['Sponsored Job?'] == 'Yes').sum()
        non_sponsored_count = total_count - sponsored_count
        sponsorship_rate = (sponsored_count / total_count) * 100 if total_count > 0 else 0
        
        roles = company_data['jobRoleName'].nunique()
        locations = company_data['location'].nunique()
        
        # Get top roles in this company
        role_counts = company_data['jobRoleName'].value_counts().head(5)
        top_roles = role_counts.to_dict()
        
        return {
            'company_name': company_name,
            'total_jobs': total_count,
            'sponsored_jobs': sponsored_count,
            'non_sponsored_jobs': non_sponsored_count,
            'sponsorship_rate': sponsorship_rate,
            'unique_roles': roles,
            'unique_locations': locations,
            'top_roles': top_roles
        }
    
    def get_top_sponsoring_roles(self, top_n: int = 20) -> pd.DataFrame:
        """Get top roles by sponsorship rate"""
        if self.df is None:
            return pd.DataFrame()
            
        role_stats = []
        for role in self.roles:
            insights = self.get_role_insights(role)
            if insights and insights['total_jobs'] >= 10:  # Only include roles with at least 10 jobs
                role_stats.append(insights)
        
        if not role_stats:
            return pd.DataFrame()
            
        df_stats = pd.DataFrame(role_stats)
        df_stats = df_stats.sort_values('sponsorship_rate', ascending=False)
        
        return df_stats.head(top_n)
    
    def get_top_sponsoring_companies(self, top_n: int = 20) -> pd.DataFrame:
        """Get top companies by sponsorship rate"""
        if self.df is None:
            return pd.DataFrame()
        
        # Filter to only companies that have at least one sponsored job
        sponsored_companies = self.df[self.df['Sponsored Job?'] == 'Yes']['company'].unique()
        
        company_stats = []
        for company in sponsored_companies:
            insights = self.get_company_insights(company)
            if insights and insights['total_jobs'] >= 5:  # Only include companies with at least 5 jobs
                company_stats.append(insights)
        
        if not company_stats:
            return pd.DataFrame()
            
        df_stats = pd.DataFrame(company_stats)
        df_stats = df_stats.sort_values('sponsorship_rate', ascending=False)
        
        return df_stats.head(top_n)

def display_overall_insights(analyzer):
    """Display overall insights with metrics and charts"""
    insights = analyzer.get_overall_insights()
    
    if not insights:
        st.error("No data available for analysis")
        return
    
    # Header
    st.markdown('<h1 class="main-header">Sponsored Jobs Analysis Engine</h1>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Jobs",
            value=f"{insights['total_jobs']:,}",
            help="Total number of job postings analyzed"
        )
    
    with col2:
        st.metric(
            label="Sponsored Jobs",
            value=f"{insights['sponsored_jobs']:,}",
            delta=f"{insights['sponsorship_rate']:.1f}%",
            help="Jobs that offer visa sponsorship"
        )
    
    with col3:
        st.metric(
            label="Non-Sponsored Jobs",
            value=f"{insights['non_sponsored_jobs']:,}",
            help="Jobs that do not offer visa sponsorship"
        )
    
    with col4:
        sponsorship_rate = insights['sponsorship_rate']
        if sponsorship_rate > 20:
            delta_color = "normal"
            help_text = "High sponsorship rate - Great opportunities!"
        elif sponsorship_rate > 10:
            delta_color = "normal"
            help_text = "Moderate sponsorship rate - Some opportunities available"
        else:
            delta_color = "inverse"
            help_text = "Low sponsorship rate - Limited opportunities"
        
        st.metric(
            label="Sponsorship Rate",
            value=f"{sponsorship_rate:.1f}%",
            help=help_text
        )
    
    # Sponsorship distribution chart
    st.subheader("Visa Sponsorship Distribution")
    
    fig = px.pie(
        values=[insights['sponsored_jobs'], insights['non_sponsored_jobs']],
        names=['Sponsored', 'Non-Sponsored'],
        color_discrete_map={'Sponsored': '#28a745', 'Non-Sponsored': '#dc3545'},
        title="Overall Visa Sponsorship Distribution"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def display_role_analysis(analyzer):
    """Display role-specific analysis"""
    st.subheader("Role-Specific Analysis")
    
    # Initialize session state for cross-filtering
    if 'role_analysis_filters' not in st.session_state:
        st.session_state.role_analysis_filters = {
            'selected_role': None,
            'selected_company': 'All',
            'sponsorship_filter': 'All'
        }
    
    # Add search functionality info
    st.info(f"**Search Tip**: Type in the dropdown to quickly find roles. Available roles: {len(analyzer.roles)}")
    
    # Role selection
    selected_role = st.selectbox(
        "Select a job role to analyze:",
        options=analyzer.roles,
        help="Choose a role to see detailed sponsorship statistics. Type to search through roles.",
        key="role_selector"
    )
    
    # Update session state
    st.session_state.role_analysis_filters['selected_role'] = selected_role
    
    if selected_role:
        insights = analyzer.get_role_insights(selected_role)
        
        if insights:
            # Role metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Jobs", f"{insights['total_jobs']:,}")
            
            with col2:
                st.metric("Sponsored Jobs", f"{insights['sponsored_jobs']:,}")
            
            with col3:
                st.metric("Sponsorship Rate", f"{insights['sponsorship_rate']:.1f}%")
            
            with col4:
                st.metric("Unique Companies", f"{insights['unique_companies']:,}")
            
            # Comparison with overall rate
            overall_insights = analyzer.get_overall_insights()
            if overall_insights:
                overall_rate = overall_insights['sponsorship_rate']
                difference = insights['sponsorship_rate'] - overall_rate
                
                if difference > 5:
                    st.success(f"This role has {difference:.1f}% HIGHER sponsorship rate than average!")
                elif difference < -5:
                    st.error(f"This role has {abs(difference):1f}% LOWER sponsorship rate than average")
                else:
                    st.info(f"This role's sponsorship rate is similar to the overall average")
            
            # Create two columns for charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Role sponsorship chart with interactivity
                fig = px.pie(
                    values=[insights['sponsored_jobs'], insights['non_sponsored_jobs']],
                    names=['Sponsored', 'Non-Sponsored'],
                    color_discrete_map={'Sponsored': '#28a745', 'Non-Sponsored': '#dc3545'},
                    title=f"Sponsorship Distribution for {selected_role}"
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True, key="role_pie_chart")
                
                # Add quick filter buttons below the chart
                st.write("**Quick Filters:**")
                col_sponsored, col_non_sponsored = st.columns(2)
                with col_sponsored:
                    if st.button("Show Sponsored Only", key="role_sponsored_btn"):
                        st.session_state.role_analysis_filters['sponsorship_filter'] = 'Sponsored Only'
                with col_non_sponsored:
                    if st.button("Show Non-Sponsored Only", key="role_non_sponsored_btn"):
                        st.session_state.role_analysis_filters['sponsorship_filter'] = 'Non-Sponsored Only'
            
            with chart_col2:
                # Top companies for this role
                role_jobs = analyzer.df[analyzer.df['jobRoleName'] == selected_role]
                company_counts = role_jobs['company'].value_counts().head(10)
                
                if not company_counts.empty:
                    fig = px.bar(
                        x=company_counts.values,
                        y=company_counts.index,
                        orientation='h',
                        title=f"Top Companies for {selected_role}",
                        color=company_counts.values,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    
                    # Display chart
                    st.plotly_chart(fig, use_container_width=True, key="role_company_chart")
                    
                    # Add quick filter buttons for top companies
                    st.write("**Quick Company Filters:**")
                    top_companies = company_counts.head(5).index.tolist()
                    if top_companies:
                        company_cols = st.columns(min(len(top_companies), 5))
                        for i, company in enumerate(top_companies):
                            with company_cols[i]:
                                if st.button(f"{company[:15]}...", key=f"role_company_btn_{i}"):
                                    st.session_state.role_analysis_filters['selected_company'] = company
            
            # Job details table for selected role
            st.subheader(f"Job Details for {selected_role}")
            
            # Filter data for selected role
            role_jobs = analyzer.df[analyzer.df['jobRoleName'] == selected_role].copy()
            
            if not role_jobs.empty:
                # Add filtering options
                st.subheader("Filter Jobs")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sponsorship status filter
                    sponsorship_options = ['All', 'Sponsored Only', 'Non-Sponsored Only']
                    sponsorship_filter = st.selectbox(
                        "Filter by Sponsorship Status:",
                        options=sponsorship_options,
                        help="Filter jobs by their sponsorship status",
                        index=sponsorship_options.index(st.session_state.role_analysis_filters.get('sponsorship_filter', 'All')),
                        key="role_sponsorship_filter"
                    )
                    st.session_state.role_analysis_filters['sponsorship_filter'] = sponsorship_filter
                
                with col2:
                    # Company filter
                    available_companies = sorted(role_jobs['company'].dropna().unique().tolist())
                    current_company = st.session_state.role_analysis_filters.get('selected_company', 'All')
                    if current_company not in ['All'] + available_companies:
                        current_company = 'All'
                    
                    company_filter = st.selectbox(
                        "Filter by Company:",
                        options=['All'] + available_companies,
                        help="Filter jobs by specific companies",
                        index=available_companies.index(current_company) + 1 if current_company != 'All' and current_company in available_companies else 0,
                        key="role_company_filter"
                    )
                    st.session_state.role_analysis_filters['selected_company'] = company_filter
                
                # Apply filters
                filtered_role_jobs = role_jobs.copy()
                
                # Apply sponsorship filter
                if sponsorship_filter == 'Sponsored Only':
                    filtered_role_jobs = filtered_role_jobs[filtered_role_jobs['Sponsored Job?'] == 'Yes']
                elif sponsorship_filter == 'Non-Sponsored Only':
                    filtered_role_jobs = filtered_role_jobs[filtered_role_jobs['Sponsored Job?'] == 'No']
                
                # Apply company filter
                if company_filter != 'All':
                    filtered_role_jobs = filtered_role_jobs[filtered_role_jobs['company'] == company_filter]
                
                # Show filter insights if any filters are applied
                if sponsorship_filter != 'All' or company_filter != 'All':
                    st.subheader("Filter Insights")
                    
                    # Calculate filtered insights
                    filtered_total = len(filtered_role_jobs)
                    filtered_sponsored = (filtered_role_jobs['Sponsored Job?'] == 'Yes').sum()
                    filtered_non_sponsored = filtered_total - filtered_sponsored
                    filtered_sponsorship_rate = (filtered_sponsored / filtered_total * 100) if filtered_total > 0 else 0
                    
                    # Calculate original role insights for comparison
                    original_total = len(role_jobs)
                    original_sponsored = (role_jobs['Sponsored Job?'] == 'Yes').sum()
                    original_sponsorship_rate = (original_sponsored / original_total * 100) if original_total > 0 else 0
                    
                    # Display comparison metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Filtered Jobs", f"{filtered_total:,}")
                    
                    with col2:
                        st.metric("Filtered Sponsored", f"{filtered_sponsored:,}")
                    
                    with col3:
                        if company_filter != 'All':
                            unique_locations = filtered_role_jobs['location'].nunique()
                            st.metric("Unique Locations", f"{unique_locations:,}")
                        else:
                            unique_companies = filtered_role_jobs['company'].nunique()
                            st.metric("Unique Companies", f"{unique_companies:,}")
                    
                    # Show insights based on filters applied
                    if company_filter != 'All':
                        # Calculate total sponsored jobs for this role
                        total_sponsored_for_role = (role_jobs['Sponsored Job?'] == 'Yes').sum()
                        
                        st.info(f"**Company Focus**: You're viewing {company_filter}'s {selected_role} positions. "
                               f"This represents {filtered_total:,} out of {original_total:,} total {selected_role} jobs "
                               f"({(filtered_total/original_total*100):.1f}% of all {selected_role} positions) and "
                               f"{filtered_sponsored:,} out of {total_sponsored_for_role:,} total sponsored {selected_role} jobs "
                               f"({(filtered_sponsored/total_sponsored_for_role*100):.1f}% of all sponsored {selected_role} positions).")
                    
                    if sponsorship_filter != 'All':
                        if sponsorship_filter == 'Sponsored Only':
                            st.success(f"**Sponsored Focus**: Showing only sponsored {selected_role} positions. "
                                     f"These {filtered_total:,} jobs represent {filtered_total/original_total*100:.1f}% of all {selected_role} positions.")
                        else:
                            st.info(f"**Non-Sponsored Focus**: Showing only non-sponsored {selected_role} positions. "
                                  f"These {filtered_total:,} jobs represent {filtered_total/original_total*100:.1f}% of all {selected_role} positions.")
                
                if not filtered_role_jobs.empty:
                    # Sort by date posted (most recent first)
                    if 'datePosted' in filtered_role_jobs.columns:
                        try:
                            filtered_role_jobs['datePosted'] = pd.to_datetime(filtered_role_jobs['datePosted'], errors='coerce')
                            filtered_role_jobs = filtered_role_jobs.sort_values('datePosted', ascending=False)
                        except:
                            filtered_role_jobs = filtered_role_jobs.sort_values('datePosted', ascending=False, na_position='last')
                    
                    # Select columns to display
                    display_columns = ['title', 'company', 'location', 'datePosted', 'url', 'Sponsored Job?']
                    available_columns = [col for col in display_columns if col in filtered_role_jobs.columns]
                    
                    # Create display dataframe
                    display_df = filtered_role_jobs[available_columns].copy()
                    
                    # Add pagination controls
                    st.subheader("Table Settings")
                    col_size, col_info = st.columns([1, 2])
                    
                    with col_size:
                        page_size = st.selectbox(
                            "Rows per page:",
                            options=[25, 50, 100, 250, 500],
                            index=0,  # Default to 25
                            key="role_page_size"
                        )
                    
                    with col_info:
                        st.info(f"Showing {len(display_df):,} jobs for {selected_role} (filtered). Links are clickable!")
                    
                    # Apply pagination
                    page_data, current_page, total_pages = paginate_dataframe(display_df, page_size, "role_page")
                    
                    # Display pagination controls
                    display_pagination_controls(current_page, total_pages, "role_page")
                    
                    # Display the table using Streamlit's native components with sorting
                    table_data = create_clickable_table(page_data)
                    st.dataframe(
                        table_data, 
                        use_container_width=True, 
                        height=400,
                        column_config={
                            "url": st.column_config.LinkColumn(
                                "url",
                                help="Click to open job posting",
                                display_text="Link"
                            )
                        }
                    )
                    
                    # Download button
                    csv = display_df.to_csv(index=False)
                    filter_suffix = f"_{sponsorship_filter.lower().replace(' ', '_')}" if sponsorship_filter != 'All' else ""
                    company_suffix = f"_{company_filter.replace(' ', '_')}" if company_filter != 'All' else ""
                    st.download_button(
                        label=f"Download {selected_role} Jobs as CSV",
                        data=csv,
                        file_name=f"{selected_role.replace(' ', '_')}_jobs{filter_suffix}{company_suffix}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No jobs match the selected filters.")
            else:
                st.warning(f"No job details found for {selected_role}")

def display_company_analysis(analyzer):
    """Display company-specific analysis"""
    st.subheader("Company-Specific Analysis")
    
    # Get companies with sponsored jobs
    sponsored_companies = analyzer.df[analyzer.df['Sponsored Job?'] == 'Yes']['company'].unique()
    company_counts = analyzer.df['company'].value_counts()
    
    # Sort by total job count and take top 200 for better searchability
    sponsored_companies_with_counts = [(company, company_counts[company]) for company in sponsored_companies]
    sponsored_companies_with_counts.sort(key=lambda x: x[1], reverse=True)
    top_sponsored_companies = [company for company, _ in sponsored_companies_with_counts[:200]]
    
    if not top_sponsored_companies:
        st.error("No companies with sponsored jobs found")
        return
    
    # Initialize session state for cross-filtering
    if 'company_analysis_filters' not in st.session_state:
        st.session_state.company_analysis_filters = {
            'selected_company': None,
            'selected_role': 'All',
            'sponsorship_filter': 'All'
        }
    
    # Add search functionality
    st.info(f"💡 **Search Tip**: Type in the dropdown to quickly find companies. Showing top {len(top_sponsored_companies)} companies with sponsored jobs.")
    
    # Company selection
    selected_company = st.selectbox(
        "Select a company to analyze:",
        options=top_sponsored_companies,
        help="Choose a company to see detailed sponsorship statistics. Type to search through companies.",
        key="company_selector"
    )
    
    # Update session state
    st.session_state.company_analysis_filters['selected_company'] = selected_company
    
    if selected_company:
        insights = analyzer.get_company_insights(selected_company)
        
        if insights:
            # Company metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Jobs", f"{insights['total_jobs']:,}")
            
            with col2:
                st.metric("Sponsored Jobs", f"{insights['sponsored_jobs']:,}")
            
            with col3:
                st.metric("Sponsorship Rate", f"{insights['sponsorship_rate']:.1f}%")
            
            with col4:
                st.metric("Unique Roles", f"{insights['unique_roles']:,}")
            
            # Top roles in this company
            if insights['top_roles']:
                st.subheader(f"Top Roles at {selected_company}")
                
                roles_df = pd.DataFrame([
                    {'Role': role, 'Job Count': count}
                    for role, count in insights['top_roles'].items()
                ])
                
                fig = px.bar(
                    roles_df,
                    x='Job Count',
                    y='Role',
                    orientation='h',
                    title=f"Top Roles at {selected_company}",
                    color='Job Count',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Create two columns for charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Company sponsorship chart with interactivity
                fig = px.pie(
                    values=[insights['sponsored_jobs'], insights['non_sponsored_jobs']],
                    names=['Sponsored', 'Non-Sponsored'],
                    color_discrete_map={'Sponsored': '#28a745', 'Non-Sponsored': '#dc3545'},
                    title=f"Sponsorship Distribution for {selected_company}"
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True, key="company_pie_chart")
                
                # Add quick filter buttons below the chart
                st.write("**Quick Filters:**")
                col_sponsored, col_non_sponsored = st.columns(2)
                with col_sponsored:
                    if st.button("Show Sponsored Only", key="company_sponsored_btn"):
                        st.session_state.company_analysis_filters['sponsorship_filter'] = 'Sponsored Only'
                with col_non_sponsored:
                    if st.button("Show Non-Sponsored Only", key="company_non_sponsored_btn"):
                        st.session_state.company_analysis_filters['sponsorship_filter'] = 'Non-Sponsored Only'
            
            with chart_col2:
                # Top roles for this company
                company_jobs = analyzer.df[analyzer.df['company'] == selected_company]
                role_counts = company_jobs['jobRoleName'].value_counts().head(10)
                
                if not role_counts.empty:
                    fig = px.bar(
                        x=role_counts.values,
                        y=role_counts.index,
                        orientation='h',
                        title=f"Top Roles at {selected_company}",
                        color=role_counts.values,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    
                    # Display chart
                    st.plotly_chart(fig, use_container_width=True, key="company_role_chart")
                    
                    # Add quick filter buttons for top roles
                    st.write("**Quick Role Filters:**")
                    top_roles = role_counts.head(5).index.tolist()
                    if top_roles:
                        role_cols = st.columns(min(len(top_roles), 5))
                        for i, role in enumerate(top_roles):
                            with role_cols[i]:
                                if st.button(f"{role[:15]}...", key=f"company_role_btn_{i}"):
                                    st.session_state.company_analysis_filters['selected_role'] = role
            
            # Job details table for selected company
            st.subheader(f"Job Details for {selected_company}")
            
            # Filter data for selected company
            company_jobs = analyzer.df[analyzer.df['company'] == selected_company].copy()
            
            if not company_jobs.empty:
                # Add filtering options
                st.subheader("Filter Jobs")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sponsorship status filter
                    sponsorship_options = ['All', 'Sponsored Only', 'Non-Sponsored Only']
                    sponsorship_filter = st.selectbox(
                        "Filter by Sponsorship Status:",
                        options=sponsorship_options,
                        help="Filter jobs by their sponsorship status",
                        index=sponsorship_options.index(st.session_state.company_analysis_filters.get('sponsorship_filter', 'All')),
                        key="company_sponsorship_filter"
                    )
                    st.session_state.company_analysis_filters['sponsorship_filter'] = sponsorship_filter
                
                with col2:
                    # Job role filter
                    available_roles = sorted(company_jobs['jobRoleName'].dropna().unique().tolist())
                    current_role = st.session_state.company_analysis_filters.get('selected_role', 'All')
                    if current_role not in ['All'] + available_roles:
                        current_role = 'All'
                    
                    role_filter = st.selectbox(
                        "Filter by Job Role:",
                        options=['All'] + available_roles,
                        help="Filter jobs by specific roles",
                        index=available_roles.index(current_role) + 1 if current_role != 'All' and current_role in available_roles else 0,
                        key="company_role_filter"
                    )
                    st.session_state.company_analysis_filters['selected_role'] = role_filter
                
                # Apply filters
                filtered_company_jobs = company_jobs.copy()
                
                # Apply sponsorship filter
                if sponsorship_filter == 'Sponsored Only':
                    filtered_company_jobs = filtered_company_jobs[filtered_company_jobs['Sponsored Job?'] == 'Yes']
                elif sponsorship_filter == 'Non-Sponsored Only':
                    filtered_company_jobs = filtered_company_jobs[filtered_company_jobs['Sponsored Job?'] == 'No']
                
                # Apply role filter
                if role_filter != 'All':
                    filtered_company_jobs = filtered_company_jobs[filtered_company_jobs['jobRoleName'] == role_filter]
                
                # Show filter insights if any filters are applied
                if sponsorship_filter != 'All' or role_filter != 'All':
                    st.subheader("Filter Insights")
                    
                    # Calculate filtered insights
                    filtered_total = len(filtered_company_jobs)
                    filtered_sponsored = (filtered_company_jobs['Sponsored Job?'] == 'Yes').sum()
                    filtered_non_sponsored = filtered_total - filtered_sponsored
                    filtered_sponsorship_rate = (filtered_sponsored / filtered_total * 100) if filtered_total > 0 else 0
                    
                    # Calculate original company insights for comparison
                    original_total = len(company_jobs)
                    original_sponsored = (company_jobs['Sponsored Job?'] == 'Yes').sum()
                    original_sponsorship_rate = (original_sponsored / original_total * 100) if original_total > 0 else 0
                    
                    # Display comparison metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Filtered Jobs", f"{filtered_total:,}")
                    
                    with col2:
                        st.metric("Filtered Sponsored", f"{filtered_sponsored:,}")
                    
                    with col3:
                        if role_filter != 'All':
                            unique_locations = filtered_company_jobs['location'].nunique()
                            st.metric("Unique Locations", f"{unique_locations:,}")
                        else:
                            unique_roles = filtered_company_jobs['jobRoleName'].nunique()
                            st.metric("Unique Roles", f"{unique_roles:,}")
                    
                    # Show insights based on filters applied
                    if role_filter != 'All':
                        # Calculate total sponsored jobs for this company
                        total_sponsored_for_company = (company_jobs['Sponsored Job?'] == 'Yes').sum()
                        
                        st.info(f"**Role Focus**: You're viewing {selected_company}'s {role_filter} positions. "
                               f"This represents {filtered_total:,} out of {original_total:,} total {selected_company} jobs "
                               f"({(filtered_total/original_total*100):.1f}% of all {selected_company} positions) and "
                               f"{filtered_sponsored:,} out of {total_sponsored_for_company:,} total sponsored {selected_company} jobs "
                               f"({(filtered_sponsored/total_sponsored_for_company*100):.1f}% of all sponsored {selected_company} positions).")
                    
                    if sponsorship_filter != 'All':
                        if sponsorship_filter == 'Sponsored Only':
                            st.success(f"**Sponsored Focus**: Showing only sponsored {selected_company} positions. "
                                     f"These {filtered_total:,} jobs represent {filtered_total/original_total*100:.1f}% of all {selected_company} positions.")
                        else:
                            st.info(f"**Non-Sponsored Focus**: Showing only non-sponsored {selected_company} positions. "
                                  f"These {filtered_total:,} jobs represent {filtered_total/original_total*100:.1f}% of all {selected_company} positions.")
                
                if not filtered_company_jobs.empty:
                    # Sort by date posted (most recent first)
                    if 'datePosted' in filtered_company_jobs.columns:
                        try:
                            filtered_company_jobs['datePosted'] = pd.to_datetime(filtered_company_jobs['datePosted'], errors='coerce')
                            filtered_company_jobs = filtered_company_jobs.sort_values('datePosted', ascending=False)
                        except:
                            filtered_company_jobs = filtered_company_jobs.sort_values('datePosted', ascending=False, na_position='last')
                    
                    # Select columns to display
                    display_columns = ['title', 'jobRoleName', 'location', 'datePosted', 'url', 'Sponsored Job?']
                    available_columns = [col for col in display_columns if col in filtered_company_jobs.columns]
                    
                    # Create display dataframe
                    display_df = filtered_company_jobs[available_columns].copy()
                    
                    # Add pagination controls
                    st.subheader("Table Settings")
                    col_size, col_info = st.columns([1, 2])
                    
                    with col_size:
                        page_size = st.selectbox(
                            "Rows per page:",
                            options=[25, 50, 100, 250, 500],
                            index=0,  # Default to 25
                            key="company_page_size"
                        )
                    
                    with col_info:
                        st.info(f"Showing {len(display_df):,} jobs for {selected_company} (filtered). Links are clickable!")
                    
                    # Apply pagination
                    page_data, current_page, total_pages = paginate_dataframe(display_df, page_size, "company_page")
                    
                    # Display pagination controls
                    display_pagination_controls(current_page, total_pages, "company_page")
                    
                    # Display the table using Streamlit's native components with sorting
                    table_data = create_clickable_table(page_data)
                    st.dataframe(
                        table_data, 
                        use_container_width=True, 
                        height=400,
                        column_config={
                            "url": st.column_config.LinkColumn(
                                "url",
                                help="Click to open job posting",
                                display_text="Link"
                            )
                        }
                    )
                    
                    # Download button
                    csv = display_df.to_csv(index=False)
                    filter_suffix = f"_{sponsorship_filter.lower().replace(' ', '_')}" if sponsorship_filter != 'All' else ""
                    role_suffix = f"_{role_filter.replace(' ', '_')}" if role_filter != 'All' else ""
                    st.download_button(
                        label=f"Download {selected_company} Jobs as CSV",
                        data=csv,
                        file_name=f"{selected_company.replace(' ', '_').replace('&', 'and')}_jobs{filter_suffix}{role_suffix}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No jobs match the selected filters.")
            else:
                st.warning(f"No job details found for {selected_company}")

def display_top_performers(analyzer):
    """Display top performing roles and companies"""
    st.subheader("Top Performers")
    
    # Top roles
    st.subheader("Top Roles by Sponsorship Rate")
    top_roles = analyzer.get_top_sponsoring_roles(15)
    
    if not top_roles.empty:
        fig = px.bar(
            top_roles,
            x='sponsorship_rate',
            y='role_name',
            orientation='h',
            title="Top 15 Roles by Sponsorship Rate (min 10 jobs)",
            labels={'sponsorship_rate': 'Sponsorship Rate (%)', 'role_name': 'Job Role'},
            color='sponsorship_rate',
            color_continuous_scale='Greens'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Top roles table
        st.subheader("Top Roles Details")
        display_df = top_roles[['role_name', 'total_jobs', 'sponsored_jobs', 'sponsorship_rate', 'unique_companies']].copy()
        display_df.columns = ['Role', 'Total Jobs', 'Sponsored Jobs', 'Sponsorship Rate (%)', 'Companies']
        display_df['Sponsorship Rate (%)'] = display_df['Sponsorship Rate (%)'].round(1)
        st.dataframe(display_df, use_container_width=True)
    
    # Top companies
    st.subheader("Top Companies by Sponsorship Rate")
    top_companies = analyzer.get_top_sponsoring_companies(15)
    
    if not top_companies.empty:
        fig = px.bar(
            top_companies,
            x='sponsorship_rate',
            y='company_name',
            orientation='h',
            title="Top 15 Companies by Sponsorship Rate (min 5 jobs)",
            labels={'sponsorship_rate': 'Sponsorship Rate (%)', 'company_name': 'Company'},
            color='sponsorship_rate',
            color_continuous_scale='Blues'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Top companies table
        st.subheader("Top Companies Details")
        display_df = top_companies[['company_name', 'total_jobs', 'sponsored_jobs', 'sponsorship_rate', 'unique_roles']].copy()
        display_df.columns = ['Company', 'Total Jobs', 'Sponsored Jobs', 'Sponsorship Rate (%)', 'Roles']
        display_df['Sponsorship Rate (%)'] = display_df['Sponsorship Rate (%)'].round(1)
        st.dataframe(display_df, use_container_width=True)

def display_data_explorer(analyzer):
    """Display interactive data explorer"""
    st.subheader("Data Explorer")
    
    # Initialize session state for cross-filtering
    if 'data_explorer_filters' not in st.session_state:
        st.session_state.data_explorer_filters = {
            'sponsorship_filter': 'All',
            'role_filter': 'All',
            'company_filter': 'All'
        }
    
    # Add search functionality info
    st.info("💡 **Search Tips**: Type in the dropdowns to quickly find roles and companies. All dropdowns are searchable!")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sponsorship_filter = st.selectbox(
            "Filter by Sponsorship:",
            options=['All', 'Sponsored Only', 'Non-Sponsored Only'],
            help="Filter jobs by sponsorship status",
            index=['All', 'Sponsored Only', 'Non-Sponsored Only'].index(st.session_state.data_explorer_filters.get('sponsorship_filter', 'All')),
            key="explorer_sponsorship_filter"
        )
        st.session_state.data_explorer_filters['sponsorship_filter'] = sponsorship_filter
    
    with col2:
        current_role = st.session_state.data_explorer_filters.get('role_filter', 'All')
        role_filter = st.selectbox(
            "Filter by Role:",
            options=['All'] + analyzer.roles,
            help="Filter jobs by specific role. Type to search through roles.",
            index=analyzer.roles.index(current_role) + 1 if current_role != 'All' and current_role in analyzer.roles else 0,
            key="explorer_role_filter"
        )
        st.session_state.data_explorer_filters['role_filter'] = role_filter
    
    with col3:
        # Get all companies with sponsored jobs for better filtering
        all_companies_with_sponsored = analyzer.df[analyzer.df['Sponsored Job?'] == 'Yes']['company'].unique()
        company_options = ['All'] + sorted(all_companies_with_sponsored.tolist())
        
        current_company = st.session_state.data_explorer_filters.get('company_filter', 'All')
        company_filter = st.selectbox(
            "Filter by Company:",
            options=company_options,
            help="Filter jobs by specific company. Type to search through companies.",
            index=company_options.index(current_company) if current_company != 'All' and current_company in company_options else 0,
            key="explorer_company_filter"
        )
        st.session_state.data_explorer_filters['company_filter'] = company_filter
    
    # Apply filters
    filtered_df = analyzer.df.copy()
    
    if sponsorship_filter == 'Sponsored Only':
        filtered_df = filtered_df[filtered_df['Sponsored Job?'] == 'Yes']
    elif sponsorship_filter == 'Non-Sponsored Only':
        filtered_df = filtered_df[filtered_df['Sponsored Job?'] == 'No']
    
    if role_filter != 'All':
        filtered_df = filtered_df[filtered_df['jobRoleName'] == role_filter]
    
    if company_filter != 'All':
        filtered_df = filtered_df[filtered_df['company'] == company_filter]
    
    # Sort by date posted in descending order (newest first)
    if 'datePosted' in filtered_df.columns:
        # Convert datePosted to datetime if it's not already
        try:
            filtered_df['datePosted'] = pd.to_datetime(filtered_df['datePosted'], errors='coerce')
            filtered_df = filtered_df.sort_values('datePosted', ascending=False)
        except:
            # If conversion fails, try to sort as string
            filtered_df = filtered_df.sort_values('datePosted', ascending=False, na_position='last')
    
    # Display filtered results
    st.subheader(f"Filtered Results ({len(filtered_df):,} jobs)")
    
    # Add interactive charts
    if len(filtered_df) > 0:
        st.subheader("Interactive Charts")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Sponsorship distribution chart
            sponsored_count = (filtered_df['Sponsored Job?'] == 'Yes').sum()
            non_sponsored_count = len(filtered_df) - sponsored_count
            
            fig = px.pie(
                values=[sponsored_count, non_sponsored_count],
                names=['Sponsored', 'Non-Sponsored'],
                color_discrete_map={'Sponsored': '#28a745', 'Non-Sponsored': '#dc3545'},
                title="Sponsorship Distribution (Filtered)"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True, key="explorer_pie_chart")
            
            # Add quick filter buttons below the chart
            st.write("**Quick Filters:**")
            col_sponsored, col_non_sponsored = st.columns(2)
            with col_sponsored:
                if st.button("Show Sponsored Only", key="explorer_sponsored_btn"):
                    st.session_state.data_explorer_filters['sponsorship_filter'] = 'Sponsored Only'
            with col_non_sponsored:
                if st.button("Show Non-Sponsored Only", key="explorer_non_sponsored_btn"):
                    st.session_state.data_explorer_filters['sponsorship_filter'] = 'Non-Sponsored Only'
        
        with chart_col2:
            # Top companies or roles chart
            if role_filter != 'All' and company_filter == 'All':
                # Show top companies for this role
                company_counts = filtered_df['company'].value_counts().head(10)
                if not company_counts.empty:
                    fig = px.bar(
                        x=company_counts.values,
                        y=company_counts.index,
                        orientation='h',
                        title=f"Top Companies for {role_filter}",
                        color=company_counts.values,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    
                    # Display chart
                    st.plotly_chart(fig, use_container_width=True, key="explorer_company_chart")
                    
                    # Add quick filter buttons for top companies
                    st.write("**Quick Company Filters:**")
                    top_companies = company_counts.head(5).index.tolist()
                    if top_companies:
                        company_cols = st.columns(min(len(top_companies), 5))
                        for i, company in enumerate(top_companies):
                            with company_cols[i]:
                                if st.button(f"{company[:15]}...", key=f"explorer_company_btn_{i}"):
                                    st.session_state.data_explorer_filters['company_filter'] = company
            
            elif company_filter != 'All' and role_filter == 'All':
                # Show top roles for this company
                role_counts = filtered_df['jobRoleName'].value_counts().head(10)
                if not role_counts.empty:
                    fig = px.bar(
                        x=role_counts.values,
                        y=role_counts.index,
                        orientation='h',
                        title=f"Top Roles at {company_filter}",
                        color=role_counts.values,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    
                    # Display chart
                    st.plotly_chart(fig, use_container_width=True, key="explorer_role_chart")
                    
                    # Add quick filter buttons for top roles
                    st.write("**Quick Role Filters:**")
                    top_roles = role_counts.head(5).index.tolist()
                    if top_roles:
                        role_cols = st.columns(min(len(top_roles), 5))
                        for i, role in enumerate(top_roles):
                            with role_cols[i]:
                                if st.button(f"{role[:15]}...", key=f"explorer_role_btn_{i}"):
                                    st.session_state.data_explorer_filters['role_filter'] = role
            
            else:
                # Show top roles overall
                role_counts = filtered_df['jobRoleName'].value_counts().head(10)
                if not role_counts.empty:
                    fig = px.bar(
                        x=role_counts.values,
                        y=role_counts.index,
                        orientation='h',
                        title="Top Roles (Filtered)",
                        color=role_counts.values,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    
                    # Display chart
                    st.plotly_chart(fig, use_container_width=True, key="explorer_role_chart")
                    
                    # Add quick filter buttons for top roles
                    st.write("**Quick Role Filters:**")
                    top_roles = role_counts.head(5).index.tolist()
                    if top_roles:
                        role_cols = st.columns(min(len(top_roles), 5))
                        for i, role in enumerate(top_roles):
                            with role_cols[i]:
                                if st.button(f"{role[:15]}...", key=f"explorer_role_btn_{i}"):
                                    st.session_state.data_explorer_filters['role_filter'] = role
    
    # Show insights if any filters are applied
    if sponsorship_filter != 'All' or role_filter != 'All' or company_filter != 'All':
        st.subheader("Filter Insights")
        
        # Calculate filtered insights
        filtered_total = len(filtered_df)
        filtered_sponsored = (filtered_df['Sponsored Job?'] == 'Yes').sum()
        filtered_non_sponsored = filtered_total - filtered_sponsored
        filtered_sponsorship_rate = (filtered_sponsored / filtered_total * 100) if filtered_total > 0 else 0
        
        # Calculate original insights for comparison
        original_total = len(analyzer.df)
        original_sponsored = (analyzer.df['Sponsored Job?'] == 'Yes').sum()
        original_sponsorship_rate = (original_sponsored / original_total * 100) if original_total > 0 else 0
        
        # Display comparison metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Filtered Jobs", f"{filtered_total:,}")
        
        with col2:
            st.metric("Filtered Sponsored", f"{filtered_sponsored:,}")
        
        with col3:
            if role_filter != 'All' and company_filter != 'All':
                unique_locations = filtered_df['location'].nunique()
                st.metric("Unique Locations", f"{unique_locations:,}")
            elif role_filter != 'All':
                unique_companies = filtered_df['company'].nunique()
                st.metric("Unique Companies", f"{unique_companies:,}")
            elif company_filter != 'All':
                unique_roles = filtered_df['jobRoleName'].nunique()
                st.metric("Unique Roles", f"{unique_roles:,}")
            else:
                unique_roles = filtered_df['jobRoleName'].nunique()
                st.metric("Unique Roles", f"{unique_roles:,}")
        
        # Show insights based on filters applied
        if role_filter != 'All' and company_filter != 'All':
            # Calculate total sponsored jobs for this role
            role_data = analyzer.df[analyzer.df['jobRoleName'] == role_filter]
            total_sponsored_for_role = (role_data['Sponsored Job?'] == 'Yes').sum()
            
            st.info(f"**Role & Company Focus**: You're viewing {company_filter}'s {role_filter} positions. "
                   f"This represents {filtered_total:,} out of {original_total:,} total jobs "
                   f"({(filtered_total/original_total*100):.1f}% of all jobs in the dataset) and "
                   f"{filtered_sponsored:,} out of {total_sponsored_for_role:,} total sponsored {role_filter} jobs "
                   f"({(filtered_sponsored/total_sponsored_for_role*100):.1f}% of all sponsored {role_filter} positions).")
        
        elif role_filter != 'All':
            # Calculate total sponsored jobs for this role
            role_data = analyzer.df[analyzer.df['jobRoleName'] == role_filter]
            total_sponsored_for_role = (role_data['Sponsored Job?'] == 'Yes').sum()
            
            st.info(f"**Role Focus**: You're viewing all {role_filter} positions. "
                   f"This represents {filtered_total:,} out of {original_total:,} total jobs "
                   f"({(filtered_total/original_total*100):.1f}% of all jobs in the dataset) and "
                   f"{filtered_sponsored:,} out of {total_sponsored_for_role:,} total sponsored {role_filter} jobs "
                   f"({(filtered_sponsored/total_sponsored_for_role*100):.1f}% of all sponsored {role_filter} positions).")
        
        elif company_filter != 'All':
            # Calculate total sponsored jobs for this company
            company_data = analyzer.df[analyzer.df['company'] == company_filter]
            total_sponsored_for_company = (company_data['Sponsored Job?'] == 'Yes').sum()
            
            st.info(f"**Company Focus**: You're viewing all {company_filter} positions. "
                   f"This represents {filtered_total:,} out of {original_total:,} total jobs "
                   f"({(filtered_total/original_total*100):.1f}% of all jobs in the dataset) and "
                   f"{filtered_sponsored:,} out of {total_sponsored_for_company:,} total sponsored {company_filter} jobs "
                   f"({(filtered_sponsored/total_sponsored_for_company*100):.1f}% of all sponsored {company_filter} positions).")
        
        if sponsorship_filter != 'All':
            if sponsorship_filter == 'Sponsored Only':
                st.success(f"**Sponsored Focus**: Showing only sponsored positions. "
                         f"These {filtered_total:,} jobs represent {filtered_total/original_total*100:.1f}% of all jobs in the dataset.")
            else:
                st.info(f"**Non-Sponsored Focus**: Showing only non-sponsored positions. "
                      f"These {filtered_total:,} jobs represent {filtered_total/original_total*100:.1f}% of all jobs in the dataset.")
    
    # Add scrolling and link instructions
    st.info("💡 **Table Navigation**: The table below is scrollable both horizontally and vertically. Use your mouse wheel or scrollbars to navigate through the data. **Click the 🔗 Link buttons to open job postings in a new tab.**")
    
    if len(filtered_df) > 0:
        # Show key columns including URL
        display_columns = ['title', 'company', 'location', 'jobRoleName', 'Sponsored Job?', 'datePosted', 'url']
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        
        # Create a copy for display to avoid modifying the original
        display_df = filtered_df[available_columns].copy()
        
        # Add pagination controls
        st.subheader("Table Settings")
        col_size, col_info = st.columns([1, 2])
        
        with col_size:
            page_size = st.selectbox(
                "Rows per page:",
                options=[25, 50, 100, 250, 500],
                index=0,  # Default to 25
                key="explorer_page_size"
            )
        
        with col_info:
            st.info(f"Showing {len(display_df):,} filtered jobs. Links are clickable!")
        
        # Apply pagination
        page_data, current_page, total_pages = paginate_dataframe(display_df, page_size, "explorer_page")
        
        # Display pagination controls
        display_pagination_controls(current_page, total_pages, "explorer_page")
        
        # Display the table using Streamlit's native components with sorting
        table_data = create_clickable_table(page_data)
        st.dataframe(
            table_data, 
            use_container_width=True, 
            height=400,
            column_config={
                "url": st.column_config.LinkColumn(
                    "url",
                    help="Click to open job posting",
                    display_text="Link"
                )
            }
        )
        
        
        # Download button - use original data without URL formatting
        csv = filtered_df[available_columns].to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_jobs_{sponsorship_filter.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No jobs match the selected filters.")

def main():
    """Main application function"""
    # Data source selection in sidebar
    st.sidebar.title("Data Configuration")
    data_source = st.sidebar.selectbox(
        "Choose Data Source:",
        ["google_drive", "postgresql"],
        format_func=lambda x: "Google Drive" if x == "google_drive" else "PostgreSQL Database",
        help="Select where to load the data from"
    )
    
    # Show current configuration
    if data_source == "google_drive":
        st.sidebar.info("🌐 **Google Drive Mode**")
        st.sidebar.info("Make sure to update the Google Drive link in the code!")
    else:
        st.sidebar.info("🗄️ **PostgreSQL Mode**")
        st.sidebar.info("Make sure to update the database connection details in the code!")
    
    # Initialize analyzer with selected data source
    analyzer = SponsoredJobsAnalyzer(data_source)
    
    # Load data
    if not analyzer.load_data():
        st.error("Failed to load data. Please check your configuration.")
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Data Source:** " + ("Google Drive" if data_source == "google_drive" else "PostgreSQL Database"))
        st.sidebar.markdown("**Status:** Data loading failed")
        
        # Show setup instructions
        if data_source == "google_drive":
            st.markdown("## 🔧 Google Drive Setup Instructions")
            st.markdown("""
            ### Step 1: Upload Your File to Google Drive
            1. Go to [Google Drive](https://drive.google.com)
            2. Upload your `H1B-Sponsored-Jobs.parquet` file
            3. Right-click the file → "Share" → "Change to anyone with the link"
            4. Copy the sharing link
            
            ### Step 2: Get the File ID
            Your sharing link will look like: `https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing`
            Copy the `FILE_ID_HERE` part.
            
            ### Step 3: Update the Code
            In `app.py`, find this line:
            ```python
            "google_drive": "https://drive.google.com/uc?export=download&id=YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
            ```
            Replace `YOUR_GOOGLE_DRIVE_FILE_ID_HERE` with your actual file ID.
            """)
        elif data_source == "postgresql":
            st.markdown("## 🔧 PostgreSQL Setup Instructions")
            st.markdown("""
            ### ✅ Database Configuration Complete
            Your PostgreSQL database is already configured with:
            - **Host**: 34.132.93.79:5432
            - **Database**: solwizz
            - **Tables**: Job and JobRole (with JOIN query)
            
            ### Database Schema
            The app uses this SQL query to fetch data:
            ```sql
            SELECT j.*, jr.id AS "jobRoleId", jr.name as "jobRoleName"
            FROM public."Job" j
            LEFT JOIN public."JobRole" jr ON j."roleId" = jr.id
            ORDER BY j.id ASC;
            ```
            
            ### Required Columns
            Your Job table should have columns like:
            - `Sponsored Job?` (text: 'Yes' or 'No')
            - `company` (text)
            - `location` (text)
            - `title` (text)
            - `url` (text)
            - `datePosted` (date/timestamp)
            - `roleId` (foreign key to JobRole table)
            
            Your JobRole table should have:
            - `id` (primary key)
            - `name` (job role name)
            
            ### Connection Status
            The app will automatically connect to your database when you select "PostgreSQL Database" as the data source.
            """)
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type:",
        [
            "Overview",
            "Role Analysis", 
            "Company Analysis",
            "Top Performers",
            "Data Explorer"
        ],
        help="Select the type of analysis you want to perform"
    )
    
    # Display selected page
    if page == "Overview":
        display_overall_insights(analyzer)
    elif page == "Role Analysis":
        display_role_analysis(analyzer)
    elif page == "Company Analysis":
        display_company_analysis(analyzer)
    elif page == "Top Performers":
        display_top_performers(analyzer)
    elif page == "Data Explorer":
        display_data_explorer(analyzer)
    
    # Footer - only show if data is loaded successfully
    if analyzer.df is not None and analyzer.roles is not None and analyzer.companies is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Data Source:** " + ("Google Drive" if data_source == "google_drive" else "PostgreSQL Database"))
        st.sidebar.markdown(f"**Total Jobs:** {len(analyzer.df):,}")
        st.sidebar.markdown(f"**Unique Roles:** {len(analyzer.roles):,}")
        st.sidebar.markdown(f"**Unique Companies:** {len(analyzer.companies):,}")
        
        # Show data source info
        if data_source == "google_drive":
            st.sidebar.markdown("**Status:** ✅ Cloud data loaded successfully")
        else:
            st.sidebar.markdown("**Status:** ✅ Database data loaded successfully")

if __name__ == "__main__":
    main()

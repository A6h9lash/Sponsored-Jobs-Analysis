# Sponsored Jobs Analysis Engine

A modern, intuitive web-based analysis tool for H1B visa sponsorship data. This application provides comprehensive insights into job sponsorship patterns, helping users understand which roles and companies are most likely to offer visa sponsorship.

## Features

### 📊 Overview Dashboard
- **Key Metrics**: Total jobs, sponsored jobs, sponsorship rates
- **Visual Charts**: Interactive pie charts and distribution graphs
- **Real-time Statistics**: Live data analysis with performance indicators

### 🎯 Role-Specific Analysis
- **Role Selection**: Dropdown interface to select any job role
- **Detailed Metrics**: Job counts, sponsorship rates, company diversity
- **Market Comparison**: Compare role performance against overall averages
- **Visual Analytics**: Interactive charts for role-specific insights

### 🏢 Company-Specific Analysis
- **Company Selection**: Choose from companies with sponsored jobs
- **Comprehensive Metrics**: Total jobs, sponsorship rates, role diversity
- **Top Roles**: See which roles are most common at each company
- **Performance Charts**: Visual representation of company sponsorship patterns

### 🏆 Top Performers
- **Top Roles**: Ranked list of roles with highest sponsorship rates
- **Top Companies**: Companies with best sponsorship track records
- **Interactive Tables**: Sortable and filterable performance data
- **Visual Rankings**: Bar charts showing top performers

### 🔍 Data Explorer
- **Advanced Filtering**: Filter by sponsorship status, role, and company
- **Interactive Data Table**: Browse and search through job data
- **Export Functionality**: Download filtered results as CSV
- **Real-time Updates**: Instant filtering and data updates

## Installation

1. **Clone or Download** the application files
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure Data File**: Make sure `H1B-Sponsored-Jobs.parquet` is in the same directory
4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Launch the App**: Run `streamlit run app.py` in your terminal
2. **Navigate**: Use the sidebar to switch between different analysis types
3. **Explore Data**: 
   - Start with the Overview to understand the big picture
   - Use Role Analysis to dive into specific job roles
   - Check Company Analysis to see which companies sponsor most
   - View Top Performers for quick insights
   - Use Data Explorer for detailed filtering and export

## Data Requirements

The application expects a parquet file (`H1B-Sponsored-Jobs.parquet`) with the following columns:
- `Sponsored Job?`: Yes/No indicating visa sponsorship
- `jobRoleName`: Job role/title
- `company`: Company name
- `location`: Job location
- `title`: Job title
- `datePosted`: When the job was posted

## Technical Features

- **Caching**: Data is cached for improved performance
- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Charts**: Built with Plotly for rich visualizations
- **Real-time Filtering**: Instant updates when filters change
- **Export Capabilities**: Download filtered data as CSV
- **Error Handling**: Graceful handling of missing data or files

## Performance

- **Optimized Loading**: Uses pandas and pyarrow for efficient data processing
- **Smart Caching**: Streamlit caching reduces load times
- **Memory Efficient**: Handles large datasets (300K+ records) smoothly
- **Fast Filtering**: Real-time filtering with minimal latency

## Browser Compatibility

- Chrome (recommended)
- Firefox
- Safari
- Edge

## Support

For issues or questions:
1. Check that the parquet file exists and has the correct format
2. Ensure all dependencies are installed correctly
3. Verify Python version compatibility (3.8+)

## Future Enhancements

- Advanced analytics and machine learning insights
- Geographic analysis and mapping
- Time-series analysis of sponsorship trends
- Integration with job posting APIs
- Custom reporting and dashboard creation

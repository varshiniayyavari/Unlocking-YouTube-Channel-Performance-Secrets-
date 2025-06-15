import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="YouTube Revenue Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    .stMetric {font-size: 18px;}
    .sidebar .sidebar-content {background-color: #f0f2f6;}
    </style>
""", unsafe_allow_html=True)

# Title and header
st.title("📊 YouTube Revenue Analytics Dashboard")
st.markdown("""
Welcome to the YouTube Revenue Analytics Dashboard! This tool analyzes video performance metrics to predict revenue using a Random Forest model. 
Explore insights, visualize trends, and make data-driven decisions for content optimization.
*Created by [Your Name] for Internship Portfolio*
""")

# File paths (default)
default_file_path = '/Users/ayyavarivarshini/Desktop/youtube_channel_real_performance_analytics.csv'
default_model_path = '/Users/ayyavarivarshini/Desktop/random_forest_revenue_model.pkl'
default_scaler_path = '/Users/ayyavarivarshini/Desktop/scaler.pkl'

# File uploader for CSV, model, and scaler
st.sidebar.header("📂 Upload Files")
uploaded_csv = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
uploaded_model = st.sidebar.file_uploader("Upload Random Forest Model File", type=["pkl"])
uploaded_scaler = st.sidebar.file_uploader("Upload Scaler File", type=["pkl"])

# Load data function
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Video Publish Time'] = pd.to_datetime(df['Video Publish Time'], errors='coerce')
    median_date = df['Video Publish Time'].dropna().median()
    df['Video Publish Time'].fillna(median_date, inplace=True)
    df['Publish Year'] = df['Video Publish Time'].dt.year
    df['Publish Month'] = df['Video Publish Time'].dt.month
    df['Publish Day'] = df['Video Publish Time'].dt.day
    df['Publish Hour'] = df['Video Publish Time'].dt.hour
    df['Revenue per View'] = df['Estimated Revenue (USD)'] / df['Views'].replace(0, np.nan)
    df['Engagement Rate'] = (df['Likes'] + df['New Comments'] + df['Shares']) / df['Views'].replace(0, np.nan)
    df['Revenue per View'].fillna(0, inplace=True)
    df['Engagement Rate'].fillna(0, inplace=True)
    return df

# Load model and scaler function
@st.cache_resource
def load_model_and_scaler(model_file, scaler_file):
    return joblib.load(model_file), joblib.load(scaler_file)

# Initialize variables
df = None
rf_model = None
scaler = None

# Load files
try:
    if uploaded_csv:
        df = load_data(uploaded_csv)
    elif os.path.exists(default_file_path):
        df = load_data(default_file_path)
    else:
        st.error(f"CSV file not found at {default_file_path}. Please upload the CSV file.")
        st.stop()

    if uploaded_model and uploaded_scaler:
        rf_model, scaler = load_model_and_scaler(uploaded_model, uploaded_scaler)
    elif os.path.exists(default_model_path) and os.path.exists(default_scaler_path):
        rf_model, scaler = load_model_and_scaler(default_model_path, default_scaler_path)
    else:
        st.error(f"Model or scaler file not found at specified paths. Please upload both files.")
        st.stop()
except Exception as e:
    st.error(f"Error loading files: {str(e)}. Please ensure the files are valid.")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")
year_options = ['All'] + sorted(df['Publish Year'].unique().tolist())
selected_year = st.sidebar.selectbox("Publish Year", year_options)
duration_range = st.sidebar.slider(
    "Video Duration (seconds)",
    min_value=int(df['Video Duration'].min()),
    max_value=int(df['Video Duration'].max()),
    value=(int(df['Video Duration'].min()), int(df['Video Duration'].max()))
)
views_range = st.sidebar.slider(
    "Views",
    min_value=int(df['Views'].min()),
    max_value=int(df['Views'].max()),
    value=(int(df['Views'].min()), int(df['Views'].max()))
)
month_options = ['All'] + sorted(df['Publish Month'].unique().tolist())
selected_month = st.sidebar.selectbox("Publish Month", month_options)

# Filter data
filtered_df = df.copy()
if selected_year != 'All':
    filtered_df = filtered_df[filtered_df['Publish Year'] == int(selected_year)]
if selected_month != 'All':
    filtered_df = filtered_df[filtered_df['Publish Month'] == int(selected_month)]
filtered_df = filtered_df[
    (filtered_df['Video Duration'] >= duration_range[0]) &
    (filtered_df['Video Duration'] <= duration_range[1]) &
    (filtered_df['Views'] >= views_range[0]) &
    (filtered_df['Views'] <= views_range[1])
]

# Key Metrics
st.header("📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Videos", len(filtered_df))
col2.metric("Total Revenue", f"${filtered_df['Estimated Revenue (USD)'].sum():,.2f}")
col3.metric("Average Views", f"{filtered_df['Views'].mean():,.0f}")
col4.metric("Avg Engagement Rate", f"{filtered_df['Engagement Rate'].mean():.3f}")

# EDA Visualizations
st.header("📊 Exploratory Data Analysis")
tab1, tab2, tab3, tab4 = st.tabs([
    "Correlation Heatmap",
    "Revenue Distribution",
    "Revenue vs Views",
    "Engagement by Publish Time"
])

with tab1:
    st.subheader("Correlation Heatmap")
    numerical_cols = ['Estimated Revenue (USD)', 'Views', 'Watch Time (hours)', 'Likes',
                      'New Comments', 'Shares', 'New Subscribers', 'Video Duration',
                      'Revenue per View', 'Engagement Rate', 'Video Thumbnail CTR (%)']
    corr_matrix = filtered_df[numerical_cols].corr()
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu',
        title='Correlation Heatmap of Key Metrics'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Revenue Distribution")
    fig = px.histogram(
        filtered_df,
        x='Estimated Revenue (USD)',
        nbins=50,
        title='Distribution of Estimated Revenue (USD)',
        marginal='rug'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Revenue vs Views")
    fig = px.scatter(
        filtered_df,
        x='Views',
        y='Estimated Revenue (USD)',
        trendline='ols',
        title='Estimated Revenue vs Views',
        hover_data=['Video Duration']
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Engagement Rate by Publish Time")
    hourly_engagement = filtered_df.groupby('Publish Hour')['Engagement Rate'].mean().reset_index()
    fig = px.line(
        hourly_engagement,
        x='Publish Hour',
        y='Engagement Rate',
        title='Average Engagement Rate by Publish Hour',
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

# Model Results
st.header("🤖 Model Results")
st.subheader("Random Forest Performance")
features = ['Video Duration', 'Views', 'Watch Time (hours)', 'Likes', 'New Comments',
            'Shares', 'New Subscribers', 'Video Thumbnail CTR (%)', 'Engagement Rate',
            'Revenue per View', 'Publish Year', 'Publish Month', 'Publish Day', 'Publish Hour']
if len(filtered_df) > 0:
    X_filtered = filtered_df[features]
    y_filtered = filtered_df['Estimated Revenue (USD)']
    X_filtered_scaled = scaler.transform(X_filtered)
    y_pred_filtered = rf_model.predict(X_filtered_scaled)
    rmse = np.sqrt(np.mean((y_filtered - y_pred_filtered) ** 2))
    st.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.3f}")
else:
    st.warning("No data available after filtering. Adjust filters to see RMSE.")

# Feature Importance
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
fig = px.bar(
    feature_importance,
    x='Importance',
    y='Feature',
    title='Feature Importance for Revenue Prediction',
    orientation='h'
)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(feature_importance.style.format({'Importance': '{:.3f}'}))

# Prediction Tool
st.header("💸 Predict Revenue")
st.markdown("Input video features to predict revenue using the trained Random Forest model.")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        duration = st.number_input("Video Duration (seconds)", min_value=0, value=300, step=10)
        views = st.number_input("Views", min_value=0, value=1000, step=100)
        watch_time = st.number_input("Watch Time (hours)", min_value=0.0, value=50.0, step=1.0)
        likes = st.number_input("Likes", min_value=0, value=50, step=5)
        comments = st.number_input("New Comments", min_value=0, value=10, step=1)
        shares = st.number_input("Shares", min_value=0, value=5, step=1)
        subscribers = st.number_input("New Subscribers", min_value=0, value=10, step=1)
    with col2:
        ctr = st.number_input("Video Thumbnail CTR (%)", min_value=0.0, value=5.0, step=0.1)
        engagement_rate = st.number_input("Engagement Rate", min_value=0.0, value=0.1, step=0.01)
        revenue_per_view = st.number_input("Revenue per View", min_value=0.0, value=0.01, step=0.001)
        year = st.number_input("Publish Year", min_value=2000, value=2023, step=1)
        month = st.number_input("Publish Month", min_value=1, max_value=12, value=6, step=1)
        day = st.number_input("Publish Day", min_value=1, max_value=31, value=15, step=1)
        hour = st.number_input("Publish Hour", min_value=0, max_value=23, value=12, step=1)
    submitted = st.form_submit_button("Predict Revenue")
    if submitted:
        input_data = np.array([[duration, views, watch_time, likes, comments, shares,
                                subscribers, ctr, engagement_rate, revenue_per_view,
                                year, month, day, hour]])
        input_scaled = scaler.transform(input_data)
        prediction = rf_model.predict(input_scaled)[0]
        st.success(f"Predicted Revenue: **${prediction:,.2f}**")

# Insights and Recommendations
st.header("🔍 Insights and Recommendations")
st.markdown("""
**Key Insights:**
- **Views and Watch Time** are the strongest drivers of revenue, highlighting the importance of maximizing viewership and retention.
- **Engagement Rate** correlates with revenue; high likes, comments, and shares boost monetization.
- **Thumbnail CTR** significantly impacts initial views, making thumbnail optimization critical.
- **Publish Timing** affects engagement; experiment with posting during peak hours (e.g., evenings).

**Recommendations:**
- Create **engaging, high-retention content** to increase watch time and views.
- Optimize **thumbnails** with bold visuals and text to improve CTR.
- Encourage **audience interaction** (likes, comments, shares) through calls-to-action.
- Schedule uploads during **high-engagement hours** (e.g., 6-9 PM) to maximize initial traction.
- Produce **longer videos** if they maintain viewer interest, as duration drives watch time.
""")

# Footer
st.markdown("---")
st.markdown("""
*Created by [Your Name] | Internship Project 2025*  
[GitHub Repository](https://github.com/your-username/youtube-revenue-dashboard) | Powered by Streamlit  
*Dataset: YouTube Channel Real Performance Analytics (~364 videos)*
""")

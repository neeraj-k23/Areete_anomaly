import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
import pytz

# Metric Functions
def compute_mean(series):
    return series.mean()

def compute_max(series):
    return series.max()

def compute_min(series):
    return series.min()

def compute_median(series):
    return series.median()

def compute_energy(series):
    return np.sum(series ** 2)

def compute_kurtosis(series):
    return series.kurtosis()

def compute_skew(series):
    return series.skew()

def compute_mean_abs_dev(series):
    return series.mad()

def compute_positive_counts(series):
    return np.sum(series > 0)

def compute_negative_counts(series):
    return np.sum(series < 0)

def compute_iqr(series):
    return series.quantile(0.75) - series.quantile(0.25)

def compute_std_dev(series):
    return series.std()

def compute_count_above_mean(series):
    return np.sum(series > series.mean())

def compute_range(series):
    return series.max() - series.min()

def compute_peak_count(series):
    return ((series > series.shift(1)) & (series > series.shift(-1))).sum()

def compute_median_abs_dev(series):
    return np.median(np.abs(series - np.median(series)))

def compute_zcr(series):
    return ((series[:-1] * series[1:]) < 0).sum()

def count_percentile_anomaly(series):
    return np.sum(series > np.percentile(series, 95))

aggregation_functions = {
    "Mean": compute_mean,
    "Max": compute_max,
    "Min": compute_min,
    "Median": compute_median,
    "Energy": compute_energy,
    "Kurtosis": compute_kurtosis,
    "Skewness": compute_skew,
    "Mean Absolute Deviation": compute_mean_abs_dev,
    "Positive Counts": compute_positive_counts,
    "Negative Counts": compute_negative_counts,
    "Interquartile Range": compute_iqr,
    "Standard Deviation": compute_std_dev,
    "Count Above Mean": compute_count_above_mean,
    "Range": compute_range,
    "Peak Count": compute_peak_count,
    "Median Absolute Deviation": compute_median_abs_dev,
    "Zero Crossing Rate": compute_zcr,
    "Count Percentile Anomaly": count_percentile_anomaly
}

# Helper Functions
def create_df(file_name):
    df = pd.read_csv(file_name)
    if df['time'].dtype == 'object':
        df['time'] = pd.to_datetime(df['time'])
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('Asia/Kolkata')  # Replace with your timezone
    return df

def filter_time_window(df, start_datetime, end_datetime):
    return df[(df['time'] >= start_datetime) & (df['time'] <= end_datetime)]

def find_closest_time(df, current_date, current_time, timezone):
    current_datetime = timezone.localize(datetime.combine(current_date, current_time))
    df['time_diff'] = abs(df['time'] - current_datetime)
    closest_time = df.loc[df['time_diff'].idxmin()]['time']
    df.drop(columns=['time_diff'], inplace=True)
    return closest_time

def filter_current_time_window(df, closest_time, window_type, window_value, selected_days=None):
    if window_type == 'short':
        end_datetime = closest_time + timedelta(hours=window_value)
        return df[(df['time'] >= closest_time) & (df['time'] <= end_datetime)]
    elif window_type == 'long':
        end_datetime = closest_time + timedelta(days=selected_days)
        return df[(df['time'] >= closest_time) & (df['time'] <= end_datetime)]

def calculate_hourly_means(df, column):
    return df.groupby(df['time'].dt.floor('H'))[column].mean()

def calculate_10min_means(df, column):
    df.set_index('time', inplace=True)
    return df.resample('10T')[column].mean().reset_index()

def group_columns(data):
    accelerometer_columns = data.filter(regex='^a').columns.tolist()
    magnetometer_columns = data.filter(regex='^m').columns.tolist()
    gyroscope_columns = data.filter(regex='^g').columns.tolist()
    sensor_columns = accelerometer_columns + magnetometer_columns + gyroscope_columns
    other_columns = [col for col in data.columns if col not in sensor_columns and col != 'time']
    return accelerometer_columns, magnetometer_columns, gyroscope_columns, other_columns

def get_axis_columns(columns, axis):
    if axis == 'res':
        return [col for col in columns if 'x' not in col and 'y' not in col and 'z' not in col]
    return [col for col in columns if axis in col]

def detect_anomalies(df, hourly_means, column):
    df = df.copy()
    df = pd.merge(df, hourly_means, on='time', how='left', suffixes=('', '_hourly_mean'))
    df['ratio'] = df[column] / df[f'{column}_hourly_mean']
    anomalies = (df['ratio'] > 1.4) | (df['ratio'] < 0.6)
    return df[anomalies]

# Streamlit App
st.title("Sensor Data Analysis")

# Load data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = create_df(uploaded_file)

    # Input date and time ranges
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    start_time = st.time_input("Start Time")
    end_time = st.time_input("End Time")

    if start_date and end_date and start_time and end_time:
        timezone = pytz.timezone('Asia/Kolkata')  # Replace with your timezone
        start_datetime = timezone.localize(datetime.combine(start_date, start_time))
        end_datetime = timezone.localize(datetime.combine(end_date, end_time))
        filtered_df = filter_time_window(df, start_datetime, end_datetime)

        # Group columns by sensor type
        accel_cols, mag_cols, gyro_cols, other_cols = group_columns(filtered_df)

        # Sidebar options to select sensor and axis
        st.sidebar.title('Select Columns')
        sensor_options = ['Accelerometer', 'Magnetometer', 'Gyroscope', 'Other']
        selected_sensor = st.sidebar.selectbox('Select Sensor', ['None'] + sensor_options)

        if selected_sensor != 'None':
            if selected_sensor == 'Accelerometer':
                selected_axis = st.sidebar.selectbox('Select Axis', ['x', 'y', 'z', 'res'])
                axis_columns = get_axis_columns(accel_cols, selected_axis)
            elif selected_sensor == 'Magnetometer':
                selected_axis = st.sidebar.selectbox('Select Axis', ['x', 'y', 'z', 'res'])
                axis_columns = get_axis_columns(mag_cols, selected_axis)
            elif selected_sensor == 'Gyroscope':
                selected_axis = st.sidebar.selectbox('Select Axis', ['x', 'y', 'z', 'res'])
                axis_columns = get_axis_columns(gyro_cols, selected_axis)
            elif selected_sensor == 'Other':
                axis_columns = other_cols

            column = st.sidebar.selectbox("Select Column", axis_columns)
            metric = st.sidebar.selectbox("Select Metric", list(aggregation_functions.keys()))

            if column and metric:
                # Input current date and time
                current_date = st.date_input("Current Date")
                current_time = st.time_input("Current Time")

                if current_date and current_time:
                    closest_time = find_closest_time(filtered_df, current_date, current_time, timezone)
                    st.write(f"Closest time in data to {current_date} {current_time}: {closest_time}")

                    # Select window type and value
                    window_type = st.radio("Select Window Type", ['short', 'long'])
                    if window_type == 'short':
                        window_value = st.selectbox("Select Number of Hours for Short Term Window", [1, 2, 3, 4, 6])
                        selected_days = None
                    elif window_type == 'long':
                        window_value = st.number_input("Select Number of Days for Long Term Window", min_value=1, max_value=365, step=1)
                        selected_days = window_value

                    if window_value:
                        fig = go.Figure()

                        # Plot Long-Term Data
                        if window_type == 'long':
                            filtered_long_df = filter_current_time_window(filtered_df, closest_time, 'long', window_value, selected_days)
                            hourly_means = calculate_hourly_means(filtered_long_df, column)
                            df_hourly_means = pd.DataFrame(hourly_means).reset_index()
                            df_hourly_means.columns = ['time', 'mean']

                            # Anomaly detection for long-term data
                            long_term_anomaly_points = detect_anomalies(filtered_long_df, df_hourly_means, column)

                            fig.add_trace(go.Scatter(x=long_term_anomaly_points['time'], y=long_term_anomaly_points[column],
                                                     mode='markers', name='Long Term Anomalies', marker=dict(color='orange')))
                            fig.add_trace(go.Scatter(x=df_hourly_means['time'], y=df_hourly_means['mean'],
                                                     mode='lines', name='Long Term Hourly Mean', line=dict(color='blue')))

                        # Plot Short-Term Data
                        if window_type == 'short':
                            filtered_short_df = filter_current_time_window(filtered_df, closest_time, 'short', window_value)
                            short_term_means = calculate_10min_means(filtered_short_df, column)

                            # Anomaly detection for short-term data
                            short_term_anomaly_points = detect_anomalies(filtered_short_df, short_term_means, column)

                            fig.add_trace(go.Scatter(x=short_term_anomaly_points['time'], y=short_term_anomaly_points[column],
                                                     mode='markers', name='Short Term Anomalies', marker=dict(color='orange')))
                            fig.add_trace(go.Scatter(x=short_term_means['time'], y=short_term_means[column],
                                                     mode='markers', name='Short Term 10 Min Mean', marker=dict(color='red')))

                        # Finalize and show the plot
                        fig.update_layout(title='Sensor Data with Long Term and Short Term Analysis',
                                          xaxis_title='Time',
                                          yaxis_title=metric,
                                          xaxis=dict(rangeslider=dict(visible=True)))

                        st.plotly_chart(fig)

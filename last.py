import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

# Define aggregation functions
def compute_mean(segment):
    return np.mean(segment)

def compute_max(segment):
    return np.max(segment)

def compute_min(segment):
    return np.min(segment)

def compute_median(segment):
    return np.median(segment)

def compute_energy(segment):
    return np.sum(np.square(segment))

def compute_kurtosis(segment):
    return kurtosis(segment)

def compute_skew(segment):
    return skew(segment)

def compute_mean_abs_dev(segment):
    return np.mean(np.abs(segment - np.mean(segment)))

def compute_positive_counts(segment):
    return np.sum(segment > 0)

def compute_negative_counts(segment):
    return np.sum(segment < 0)

def compute_iqr(segment):
    return np.percentile(segment, 75) - np.percentile(segment, 25)

def compute_std_dev(segment):
    return np.std(segment)

def compute_count_above_mean(segment):
    return np.sum(segment > np.mean(segment))

def compute_range(segment):
    return np.max(segment) - np.min(segment)

def compute_peak_count(segment):
    return np.sum((segment[1:-1] > segment[:-2]) & (segment[1:-1] > segment[2:]))

def compute_median_abs_dev(segment):
    return np.median(np.abs(segment - np.median(segment)))

def compute_zcr(segment):
    return np.sum(np.diff(np.sign(segment)) != 0)

def count_percentile_anomaly(segment, top_5_percentile, bottom_5_percentile):
    count_above = np.sum(segment > top_5_percentile)
    count_below = np.sum(segment < bottom_5_percentile)
    return count_above + count_below

# Define aggregation function dictionary
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

def create_df(file_name):
    df = pd.read_csv(file_name)
    return df

def filter_time_window(df, start_datetime, end_datetime):
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.tz_localize(None)
    filtered_df = df[(df['time'] >= start_datetime) & (df['time'] <= end_datetime)]
    return filtered_df

def find_closest_time(df, current_date, current_time):
    current_datetime = datetime.combine(current_date, current_time)
    df['time_diff'] = abs(df['time'] - current_datetime)
    closest_time = df.loc[df['time_diff'].idxmin()]['time']
    df.drop(columns=['time_diff'], inplace=True)
    return closest_time

def filter_current_time_window(df, closest_time, window_type, window_value, selected_days=None):
    if window_type == 'short':
        time_delta = timedelta(hours=window_value)
        start_datetime = closest_time - time_delta
        filtered_df = df[(df['time'] >= start_datetime) & (df['time'] <= closest_time)]
    elif window_type == 'long':
        if selected_days is None:
            raise ValueError("Selected days must be provided for long time window.")
        start_datetime = closest_time - timedelta(days=selected_days)
        filtered_df = df[(df['time'] >= start_datetime) & (df['time'] <= closest_time)]

    return filtered_df

def calculate_hourly_means(df, column):
    hourly_means = df.groupby(df['time'].dt.hour)[column].mean()
    return hourly_means

def process_metric_streams(df, func, col):
    metric_series = []
    if func == count_percentile_anomaly:
        col_data = df[col].fillna(0)
        top_5_percentile = np.percentile(col_data, 95)
        bottom_5_percentile = np.percentile(col_data, 5)
    for i in range(0, len(df), 52):
        segment = df[col].iloc[i:i+52].fillna(0)
        if func == count_percentile_anomaly:
            metric_value = func(segment, top_5_percentile, bottom_5_percentile)
        else:
            metric_value = func(segment)
        metric_series.append(metric_value)
    return metric_series

def group_columns(data):
    accelerometer_columns = data.filter(regex='^a').columns.tolist()
    magnetometer_columns = data.filter(regex='^m').columns.tolist()
    gyroscope_columns = data.filter(regex='^g').columns.tolist()
    sensor_columns = accelerometer_columns + magnetometer_columns + gyroscope_columns
    other_columns = [col for col in data.columns if col not in sensor_columns and col != 'local_time']
    return accelerometer_columns, magnetometer_columns, gyroscope_columns, other_columns

# Function to extract columns by axis
def get_axis_columns(columns, axis):
    if axis == 'res':
        return [col for col in columns if 'x' not in col and 'y' not in col and 'z' not in col]
    return [col for col in columns if axis in col]

# Streamlit app
st.title("Sensor Data Analysis")

# Load data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = create_df(uploaded_file)
    df['time'] = pd.to_datetime(df['time'])

    # Input date and time ranges
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    start_time = st.time_input("Start Time")
    end_time = st.time_input("End Time")

    if start_date and end_date and start_time and end_time:
        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)
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

            if column:
                # Select metric
                metric = st.sidebar.selectbox("Select Metric", list(aggregation_functions.keys()))

                if metric:
                    metric_func = aggregation_functions[metric]
                    metric_values = process_metric_streams(filtered_df, metric_func, column)
                    filtered_df = filtered_df.iloc[:len(metric_values)]
                    X1 = pd.DataFrame({'time': filtered_df['time'], 'metric_values': metric_values})

                    # Input current date and time
                    current_date = st.date_input("Current Date")
                    current_time = st.time_input("Current Time")

                    if current_date and current_time:
                        closest_time = find_closest_time(filtered_df, current_date, current_time)

                        st.write(f"Closest time in data to {current_date} {current_time}: {closest_time}")

                        # Select window type and value
                        window_type = st.radio("Select Window Type", ['short', 'long'])
                        if window_type == 'short':
                            window_value = st.selectbox("Select Hour Window", [1, 2, 3, 4, 6])
                            selected_days = None
                        elif window_type == 'long':
                            window_value = st.selectbox("Select Hour Window", [1, 2, 3, 4, 6])
                            selected_days = st.number_input("Select Number of Days", min_value=1, max_value=22, step=1)

                        if window_value:
                            filtered_current_df = filter_current_time_window(filtered_df, closest_time, window_type, window_value, selected_days)

                            # Calculate hourly mean values
                            hourly_means = calculate_hourly_means(filtered_current_df, column)

                            # Calculate the threshold ratio and plot
                            threshold = st.selectbox("Select Threshold", [1.1, 1.2, 1.3, 1.4])
                            ratio_values = filtered_current_df[column] / filtered_current_df['time'].dt.hour.map(hourly_means)

                            # Plot the ratio values with annotated anomalies
                            fig, ax = plt.subplots()
                            ax.plot(filtered_current_df['time'], ratio_values, label='Ratio Values')

                            # Highlight anomalies
                            anomalies = ratio_values[ratio_values > threshold]
                            ax.scatter(filtered_current_df.loc[anomalies.index, 'time'], anomalies, color='red', label='Anomalies')

                            ax.set_title('Ratio Values Over Time with Anomalies')
                            ax.set_xlabel('Date and Time')
                            ax.set_ylabel('Ratio')
                            ax.legend()
                            plt.xticks(rotation=45)
                            st.pyplot(fig)

                            st.write(f"Anomalies detected: {len(anomalies)}")

                            # Plotting other suggestions
                            fig, ax = plt.subplots()
                            ax.plot(filtered_current_df['time'], filtered_current_df[column], label='Sensor Data')
                            ax.plot(filtered_current_df.loc[anomalies.index, 'time'], filtered_current_df.loc[anomalies.index, column], 'ro', label='Anomalies')
                            ax.set_title('Sensor Data with Anomalies')
                            ax.set_xlabel('Time')
                            ax.set_ylabel('Sensor Value')
                            ax.legend()
                            st.pyplot(fig)

                            fig, ax = plt.subplots()
                            ax.hist(metric_values, bins=30, alpha=0.7, label='Metric Values')
                            ax.axvline(x=np.percentile(metric_values, 95), color='r', linestyle='--', label='95th Percentile')
                            ax.axvline(x=np.percentile(metric_values, 5), color='b', linestyle='--', label='5th Percentile')
                            ax.set_title('Histogram of Metric Values')
                            ax.set_xlabel('Metric Value')
                            ax.set_ylabel('Frequency')
                            ax.legend()
                            st.pyplot(fig)

                            fig, ax = plt.subplots()
                            ax.boxplot([filtered_current_df.loc[filtered_current_df['time'].dt.hour == hour, column] for hour in filtered_current_df['time'].dt.hour.unique()])
                            ax.set_xticklabels(filtered_current_df['time'].dt.hour.unique())
                            ax.set_title('Box Plot of Metric Values by Hour')
                            ax.set_xlabel('Hour')
                            ax.set_ylabel('Metric Value')
                            st.pyplot(fig)

                            st.write("Temporal Analysis Completed")

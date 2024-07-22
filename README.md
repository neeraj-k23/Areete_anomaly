# Areete_anomaly

# Install the necessary libraries (if not already installed):
pip install streamlit pandas numpy openpyxl pytz

# We have included two types of files process.py and other python files.

**The process file takes in a raw data csv file as an input and returns a preprocessed file with :**

1. *create_df*: Reads the sensor data from an Excel file and returns a DataFrame.
2. *df_splitter*: Splits the 'sensor_data' column in the DataFrame into individual columns.
3. *break_sensor_data*: Extracts the sensor data (acceleration, gyroscope, and magnetometer) from the split DataFrame.
4. *calc_resultant*: Calculates the resultant magnitude of the sensor data.
5. *break_cols*: Breaks the sensor data columns into individual columns.
6. *transform*: Transforms the magnetometer data to calculate the angles.
7. *get_X*: Combines the preprocessed sensor data into a single DataFrame.
8. *preprocess1*: Applies the above preprocessing steps to the input DataFrame.
9. *preprocess2*: Creates a new DataFrame with the resultant magnitudes of the sensor data.
10. *preprocess3*: Applies additional transformations like first-order difference and Fast Fourier Transform (FFT) to the preprocessed data.

***Time-Series Processing***
11. *process_with_time*: Assigns timestamps to the preprocessed data and categorizes the time of day.

# Ratio.py

Anomaly Detection Algorithm
The application uses a ratio-based anomaly detection algorithm. Anomalies are identified if the ratio of the current value to the hourly mean exceeds a threshold of 1.4 or falls below 0.6. These points are highlighted on the graph for easy identification.
Using the App
Upload CSV File:

Click on the "Upload your CSV file" button.
Select your CSV file containing the sensor data.
Filter Data by Date and Time:

Use the date and time pickers to specify the start and end date and time for the data you want to analyze.
Select Sensor and Axis:

In the sidebar, select the sensor type (Accelerometer, Magnetometer, Gyroscope, Other).
Select the axis (x, y, z, res) for the selected sensor type.
Choose the specific column for analysis from the dropdown.
Select Metric:

Choose the metric you want to analyze (Mean, Max, Min, etc.).
Specify Current Date and Time:

Input the current date and time to find the closest time in the data.
Choose Analysis Window:

Select either a short-term (hourly) or long-term (daily) analysis window.
Specify the duration for the selected window type.
View Results:

The app will display an interactive Plotly chart with the analyzed data.
Anomalies will be marked on the graph based on the specified thresholds.
Code Explanation
Metric Functions
A set of functions to calculate various metrics from the sensor data, such as mean, max, min, energy, etc.

Helper Functions
create_df: Reads the CSV file, converts the 'time' column to datetime, and localizes it to the 'Asia/Kolkata' timezone.
filter_time_window: Filters the data within the specified start and end datetime range.
find_closest_time: Finds the closest time in the dataset to a specified current date and time.
filter_current_time_window: Filters the data within the short-term or long-term window relative to the closest time.
calculate_hourly_means: Calculates the hourly mean values for the specified column.
calculate_10min_means: Calculates the mean values for 10-minute intervals.
group_columns: Groups the columns into accelerometer, magnetometer, gyroscope, and other columns.
get_axis_columns: Retrieves the columns for the specified axis (x, y, z, or res).
detect_anomalies: Detects anomalies based on the ratio of the current value to the hourly mean.
Streamlit App
File Upload: Allows users to upload their CSV file.
Date and Time Inputs: Users can specify the start and end date and time for filtering the data.
Sidebar Options: Users can select the sensor type, axis, and column for analysis. They can also choose the metric for analysis.
Anomaly Detection: The app performs anomaly detection based on the ratio of current values to hourly means and plots these anomalies on the graph.
Plotting: The app plots the long-term and short-term data along with the detected anomalies using Plotly.

# anom.py
Similar code with anomaly detection algorithm based on 95th and 5th percentile measure.

# Last.py
Overview
This application, built with Streamlit, analyzes sensor data from CSV files to detect anomalies based on various metrics. The tool allows users to upload sensor data, filter it based on date and time ranges, select specific sensors and axes, and apply different aggregation functions to detect anomalies. The results are visualized to help users identify and interpret anomalies in their sensor data.

***Features***
Data Upload: Upload sensor data in CSV format.
Date and Time Filtering: Filter data based on user-selected start and end dates and times.
Sensor Selection: Choose between accelerometer, magnetometer, gyroscope, or other sensors.
Axis Selection: Select specific axes (x, y, z, or res) for accelerometer, magnetometer, and gyroscope.
Metric Calculation: Apply various aggregation functions to the data.
Anomaly Detection: Detect anomalies based on the ratio of current sensor values to historical hourly means.
Visualization: Visualize data, anomalies, and metrics through plots.
Requirements
Python 3.7+
Streamlit
Pandas
NumPy
Matplotlib
SciPy


Upload CSV File:

Click on "Browse files" to upload your sensor data CSV file.
Set Date and Time Filters:

Select the start and end dates and times for filtering the data.
Choose Sensor and Axis:

Select the sensor type (Accelerometer, Magnetometer, Gyroscope, or Other).
Select the specific axis (x, y, z, or res).
Select Column and Metric:

Choose the column to analyze from the selected axis.
Select the metric to calculate (Mean, Max, Min, Median, Energy, Kurtosis, Skewness, etc.).
Input Current Date and Time:

Input the current date and time to find the closest data point in the filtered data.
Select Window Type and Value:

Choose between short and long window types.
For short windows, select the hour window (1, 2, 3, 4, 6).
For long windows, select the hour window and number of days.

Anomaly Detection:

Set the threshold for anomaly detection (1.1, 1.2, 1.3, 1.4).
View the plots showing ratio values over time and anomalies.

**Aggregation Functions**
Mean: Calculate the average value.
Max: Find the maximum value.
Min: Find the minimum value.
Median: Calculate the median value.
Energy: Sum of squares of values.
Kurtosis: Measure the tailedness of the distribution.
Skewness: Measure the asymmetry of the distribution.
Mean Absolute Deviation: Mean of absolute deviations from the mean.
Positive Counts: Count of positive values.
Negative Counts: Count of negative values.
Interquartile Range (IQR): Difference between 75th and 25th percentiles.
Standard Deviation: Measure of the spread of values.
Count Above Mean: Count of values above the mean.
Range: Difference between maximum and minimum values.
Peak Count: Number of local peaks.
Median Absolute Deviation: Median of absolute deviations from the median.
Zero Crossing Rate: Rate of sign changes in the series.
Count Percentile Anomaly: Count of values outside the 5th and 95th percentiles.

**Visualization**
Ratio Values Over Time: Plot showing ratio values with anomalies highlighted.
Sensor Data with Anomalies: Plot of sensor data with anomalies marked.
Histogram of Metric Values: Histogram showing the distribution of metric values.
Box Plot by Hour: Box plot of metric values grouped by hour.

# Running streamlit interface
Save python file such that your directories match.

Run command in the terminal :

streamlit run file_name.py

# Streamlit
This interface can be used for visualization of any developed anomaly detection algorithm by substituting the required code.

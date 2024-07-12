# Areete_anomaly

# Install the necessary libraries (if not already installed):
pip install streamlit pandas numpy openpyxl pytz

# We have two files included preprocess1.py and last.py:

# The preprocess one file takes in a raw data csv file as an input and returns a preprocessed file with :
# ***Data Preprocessing***
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



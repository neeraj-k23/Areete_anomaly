import streamlit as st
import pandas as pd
import numpy as np
import pytz

# Define the column list
colo = ['a_res', 'ax', 'ay', 'az', 'g_res', 'gx', 'gy', 'gz', 'm_res', 'mx', 'my', 'mz']

# Helper functions for data preparation
def create_df(file):
    df = pd.read_csv(file)
    return df

def df_splitter(df):
    df_split = df['sensor_data'].str.split(',', expand=True)
    df_split.columns = [f"col{i}" for i in range(506)]
    df_split.drop(columns=[f"col{i}" for i in range(11, 38)], inplace=True)
    return df_split

def break_sensor_data(begin, df):
    data = df
    cols0 = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']
    for x in cols0:
        data[x] = data.iloc[:, [*range(begin, data.shape[1], 9)]].values.astype(float).tolist()
        begin += 1
    return data

def calc_resultant(x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return (x**2 + y**2 + z**2)**0.5

def break_cols(data, numcol, cols):
    result = pd.DataFrame(index=data.index)
    for c in cols:
        temp = pd.DataFrame(data[c].tolist())
        temp.columns = [c + '_' + str(r) for r in range(numcol)]
        result = pd.concat([result, temp], axis=1)
    return result

def deg(x, y):
    angle_deg = []
    for i in range(len(x)):
        angle_rad = np.arctan2(x[i], y[i])
        angle_deg_single = np.degrees(angle_rad)
        angle_deg.append(np.round(angle_deg_single, 2))
    return angle_deg

def transform(data):
    assert isinstance(data, pd.DataFrame), "Data passed to transform is not a DataFrame"
    mx_cols = [col for col in data.columns if 'mx' in col]
    my_cols = [col for col in data.columns if 'my' in col]
    mz_cols = [col for col in data.columns if 'mz' in col]

    for mx_col, my_col, mz_col in zip(mx_cols, my_cols, mz_cols):
        data[f'{mx_col}_1'] = deg(data[my_col], data[mz_col])
        data[f'{my_col}_1'] = deg(data[mz_col], data[mx_col])
        data[f'{mz_col}_1'] = deg(data[mx_col], data[my_col])

    return data

def get_X(dat):
    data_sensor = dat.copy()
    data_sensor = data_sensor.applymap(lambda x: np.array(x))
    data_sensor["a_res"] = calc_resultant(data_sensor['ax'], data_sensor['ay'], data_sensor['az'])
    data_sensor["g_res"] = calc_resultant(data_sensor['gx'], data_sensor['gy'], data_sensor['gz'])
    data_sensor["m_res"] = calc_resultant(data_sensor['mx'], data_sensor['my'], data_sensor['mz'])
    X = break_cols(data_sensor, 52, ['a_res', 'ax', 'ay', 'az', 'g_res', 'gx', 'gy', 'gz', 'm_res', 'mx', 'my', 'mz'])
    assert isinstance(X, pd.DataFrame), "X is not a DataFrame"
    return X

def create_res_df(prefix, df):
    cols = [col for col in df.columns if col.startswith(prefix)]
    return pd.DataFrame(df[cols].values.reshape(-1, 1), columns=[prefix])

def preprocess1(df):
    df_split = df_splitter(df)
    df_split = break_sensor_data(11, df_split)
    df_split['utc_time'] = pd.to_datetime(pd.to_numeric(df_split.col5), unit='s')
    local_timezone = pytz.timezone('Asia/Kolkata')

    def convert_utc_to_local(utc_time):
        utc_time = utc_time.replace(tzinfo=pytz.utc)
        local_time = utc_time.astimezone(local_timezone)
        return local_time

    df_split['local_time'] = df_split['utc_time'].apply(convert_utc_to_local)
    X = get_X(df_split)
    assert isinstance(X, pd.DataFrame), "X returned to preprocess1 is not a DataFrame"
    X1 = transform(X)
    X1['time'] = df_split['local_time']
    return X1, X.shape[0]

def preprocess2(X1):
    dfs = []
    for col in colo:
        dfs.append(create_res_df(col, X1))
    merged_df = pd.concat(dfs, axis=1)
    merged_df.dropna(inplace=True)
    return merged_df

def preprocess3(merged_df):
    new_df = merged_df.copy()
    new_df = first_diff(new_df)
    new_df = ffts(new_df)
    return new_df

def first_diff(new_df):
    new_df_diff = pd.DataFrame()
    for col in new_df.columns:
        diff_list = []
        for i in range(0, len(new_df), 52):
            segment_diff = new_df[col].iloc[i:i+52].diff().fillna(0)
            diff_list.extend(segment_diff.tolist())
        new_df_diff[col] = diff_list
    new_df_diff.columns = [col + '_diff' for col in new_df_diff.columns]
    new_df = pd.concat([new_df, new_df_diff], axis=1)
    return new_df

def compute_fft_log(segment):
    fft_result = np.fft.fft(segment)
    magnitudes = np.abs(fft_result)
    log_magnitudes = np.log(magnitudes + 1e-8)
    return log_magnitudes

def ffts(new_df):
    new_df_fft_log = pd.DataFrame()
    for col in new_df.columns:
        fft_log_list = []
        for i in range(0, len(new_df), 52):
            segment = new_df[col].iloc[i:i+52].fillna(0)
            log_magnitudes = compute_fft_log(segment)
            fft_log_list.extend(log_magnitudes)
        new_df_fft_log[col] = fft_log_list
    new_df_fft_log.columns = [col + '_fft' for col in new_df_fft_log.columns]
    new_df = pd.concat([new_df, new_df_fft_log], axis=1)
    return new_df

def process_with_time(new_df, X1):
    X1 = X1[['time']]
    X1["time_of_day"] = X1["time"].apply(categorize_time_of_day)
    times = []
    for i in range(0, len(X1)):
        for j in range(3, 55):
            times.append(pd.to_datetime(X1['time'].iloc[i]) + pd.to_timedelta(j, unit='s'))
    if len(times) == new_df.shape[0]:
        new_df['time'] = times
    else:
        st.error("Error: Length of 'times' does not match the number of rows in 'new_df'")
    return new_df, X1

def categorize_time_of_day(time):
    if time.hour < 12:
        return 'morning'
    elif 12 <= time.hour < 18:
        return 'afternoon'
    else:
        return 'evening'

def adjust_time_difference(X1):
    X1_new = X1.copy()
    for i in range(1, len(X1_new)):
        previous_time = X1_new.loc[i - 1, 'time']
        current_time = X1_new.loc[i, 'time']
        time_difference = current_time - previous_time
        if time_difference != pd.Timedelta(minutes=10):
            X1_new.loc[i, 'time'] = previous_time + pd.Timedelta(minutes=10)
    return X1_new

def process_file(file):
    df = create_df(file)
    X1, _ = preprocess1(df)
    merged_df = preprocess2(X1)
    processed_df = preprocess3(merged_df)
    final_df, X1 = process_with_time(processed_df, X1)
    X1_adjusted = adjust_time_difference(X1)
    return final_df, X1_adjusted

# Streamlit code
st.title("Sensor Data Processing Application")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.write("Processing the file...")
    final_df, X1_adjusted = process_file(uploaded_file)
    st.write("File processed successfully!")

    st.write("Download the processed file:")
    st.download_button(
        label="Download CSV",
        data=final_df.to_csv(index=False).encode('utf-8'),
        file_name='processed_output.csv',
        mime='text/csv'
    )

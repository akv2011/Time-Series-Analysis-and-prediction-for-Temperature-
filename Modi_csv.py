

import pandas as pd


# load the dataset
df = pd.read_csv('/home/systemx86/Desktop/Hack/Spy_Zen_aws/Hack_earth/dataset/New_data/test_city_1.csv')
print(df.dtypes)

# Convert to numeric with errors='coerce' to handle any remaining invalid values
df['SO2_concentration'] = pd.to_numeric(df['SO2_concentration'], errors='coerce')
df['O3_concentration'] = pd.to_numeric(df['O3_concentration'], errors='coerce')
df['CO_concentration'] = pd.to_numeric(df['CO_concentration'], errors='coerce')

# Convert the Datetime column to datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%y %H:%M', errors='coerce')


# Check the updated data types
print(df.dtypes)


print ("Rows     : " ,df.shape[0])
print ("Columns  : " ,df.shape[1])
print ("\nFeatures : \n" ,df.columns.tolist())
print ("\nMissing values :  ", df.isnull().any())
print ("\nUnique values :  \n",df.nunique())
     



df.isnull().sum()

df.query('Datetime != Datetime ')



# Sample data for illustration
# Assume `df` is your DataFrame with the `NaT` values in the `Datetime` column

# Filter rows with missing 'Datetime'
missing_dates_df = df[df['Datetime'].isna()]

# Create lists to store results
before_dates = []
after_dates = []

# Loop through each index with missing 'Datetime'
for idx in missing_dates_df.index:
    # Get previous 'Datetime' if exists
    if idx > 0:
        before_date = df.loc[idx - 1, 'Datetime']
    else:
        before_date = None

    # Get next 'Datetime' if exists
    if idx < len(df) - 1:
        after_date = df.loc[idx + 1, 'Datetime']
    else:
        after_date = None

    # Append results
    before_dates.append(before_date)
    after_dates.append(after_date)

# Add the results to the missing_dates_df
missing_dates_df = missing_dates_df.copy()
missing_dates_df['Before_Datetime'] = before_dates
missing_dates_df['After_Datetime'] = after_dates

print(missing_dates_df[['ID', 'Datetime', 'Before_Datetime', 'After_Datetime']])


# Set the time increment
time_increment = pd.Timedelta(hours=1)

# Fill NaT values
for i in range(1, len(df)):
    if pd.isna(df.loc[i, 'Datetime']):
        df.loc[i, 'Datetime'] = df.loc[i - 1, 'Datetime'] + time_increment

print(df)



# Strip any leading or trailing whitespace from column names
df.columns =df.columns.str.strip()

# Now proceed with filling missing values as intended
df['CO_concentration'] = df['CO_concentration'].fillna(method='ffill')
df['Anonymous_X1'] = df['Anonymous_X1'].fillna(method='ffill')

# Apply spline interpolation for 'Particulate_matter' and 'O3_concentration'
df['Particulate_matter'] = df['Particulate_matter'].interpolate(method='linear')
df['O3_concentration'] = df['O3_concentration'].interpolate(method='linear')

# Check for any remaining missing values
print(df.isnull().sum())

df.info()




# Save the modified DataFrame to a CSV file
df.to_csv('test_modi_c1.csv', index=False)


d3=pd.read_csv('/home/systemx86/Desktop/Hack/Spy_Zen_aws/Hack_earth/test_modi_c1.csv')
d3.info()

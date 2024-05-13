import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import date, timedelta

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    start_date = st.date_input('Enter the FROM DATE: ', date.today() + timedelta(days=1))
    end_date = st.date_input('Enter the TO DATE: ', date.today() + timedelta(days=90))
    daily_spend = st.number_input('Enter the daily SPEND ($): ', value= 15000, step=1000)
    cpi = st.number_input('Enter the daily CPI ($): ',  value= 1.6, step=0.2 )
    price = st.number_input('Enter the daily PRICE ($): ', value= 6.99, step= 0.5 )
    st.selectbox('Select SUBSCRIPTION TYPE', ["WEEKLY", "YEARLY"])

    st.markdown("<h1 style='text-align: left; font-size: 16px;'>Enter Retention Rates: </h1>", unsafe_allow_html=True)

    dfrr = pd.DataFrame(
        [
            {"Renewal times": "1",  "Rate": 0.053},
            {"Renewal times": "2",  "Rate": 0.043},
            {"Renewal times": "3",  "Rate": 0.036},
            {"Renewal times": "4",  "Rate": 0.026},
            {"Renewal times": "5",  "Rate": 0.023},
            {"Renewal times": "6",  "Rate": 0.021},
            {"Renewal times": "7",  "Rate": 0.02},
            {"Renewal times": "8",  "Rate": 0.018},
            {"Renewal times": "9",  "Rate": 0.017},
            {"Renewal times": "10", "Rate": 0.015},
            {"Renewal times": "11", "Rate": 0.015},
            {"Renewal times": "12", "Rate": 0.015}
        ]
    )
    edited_df = st.data_editor(dfrr, num_rows="dynamic")

    if 'edited_df' not in st.session_state:
        st.session_state.edited_df = dfrr.copy()

def define_date(start_date, end_date, daily_spend, cpi, price):
    date_range = pd.date_range(start=start_date, end=end_date)
    df = pd.DataFrame(date_range, columns=['Date'])
    df["install_date"] = df["Date"]
    df["spend"] = daily_spend
    df["cpi"] = cpi
    df["num_install"] = df["spend"] / df["cpi"]
    df["price"] = price
    return df

def define_renew(df, *retention_rates):
    for i, rr in enumerate(retention_rates, start=1):
        df[f"rr{i}"] = rr
        df[f"remaining_user_ after_rr{i}"] = rr * df["num_install"]
        renew_offset = 10 if i == 1 else 10 + 7 * (i - 1)
        df[f"renew_{i}_at"] = df["install_date"] + pd.Timedelta(days=renew_offset)


retention_rates = edited_df.iloc[:, 1].tolist()
df = define_date(start_date, end_date, daily_spend, cpi, price)
define_renew(df, *retention_rates)

def create_combined_dataframe(df):

  # Define columns
  columns = ["Date", "install_date", "type", "num", "values"]
  
  # Create empty DataFrames for installations and spend
  df_install = pd.DataFrame(columns=columns)
  df_spend = pd.DataFrame(columns=columns)

  # Populate install data
  df_install["Date"] = df["Date"]
  df_install["install_date"] = df["install_date"]
  df_install["type"] = "install"
  df_install["num"] = df["num_install"]
  df_install["values"] = 0

  # Populate spend data
  df_spend["Date"] = df["Date"]
  df_spend["install_date"] = df["install_date"]
  df_spend["type"] = "spend"
  df_spend["num"] = 0
  df_spend["values"] = df["spend"]

  # Combine
  df_to_calculate = pd.concat([df_install, df_spend], ignore_index=True)

  # Get renew and remaining user columns
  renew_at_columns = [col for col in df.columns if col.startswith("renew_") and col.endswith("_at")]
  remaining_user_columns = [col for col in df.columns if col.startswith("remaining_user_")]

  # Iterate and process renewals
  for renew_col, remaining_col in zip(renew_at_columns, remaining_user_columns):
    df_renew = pd.DataFrame(columns=columns)
    df_renew["Date"] = df[renew_col]
    df_renew["install_date"] = df["install_date"]

    type_suffix = renew_col.split("_")[1]
    df_renew["type"] = f"renew{type_suffix}"

    df_renew["num"] = df[remaining_col]
    df_renew["values"] = df[remaining_col] * price  

    # Append renewal data
    df_to_calculate = pd.concat([df_to_calculate, df_renew], ignore_index=True)

  return df_to_calculate

df_to_calculate = create_combined_dataframe(df)

def remove_digits(text):
    return ''.join(char for char in text if not char.isdigit())
df_to_calculate['type_group'] = df_to_calculate['type'].apply(remove_digits)
df_to_calculate["Date"] =pd.to_datetime(df_to_calculate["Date"])
df_to_plot = df_to_calculate[df_to_calculate.type_group != "install"] 
df_to_plot = df_to_plot.groupby(["Date","type_group"])["values"].sum().reset_index()

# df_to_plot

#calculate when break - even
df_filtered = df_to_plot[df_to_plot['type_group'] == 'renew']
df_merged = df_filtered.merge(df_to_plot[df_to_plot['type_group'] == 'spend'], how='left', on='Date')
first_date = df_merged[df_merged['values_x'] >= df_merged['values_y']]['Date'].iloc[0]
first_date_final = first_date.strftime('%Y-%m-%d')
# print(f"First date where revenue is equal to or greater than spend: {first_date.strftime('%Y-%m-%d')}")
minus_rev_daily = df_merged[df_merged['values_x'] < df_merged['values_y']]
negative_earnings_period = len(minus_rev_daily)
total_spend_to_profit = minus_rev_daily.values_y.sum()

#metric
#metric tab
col1, col2, col3, col4 = st.columns(4)
col1.metric("Break-even (daily): ", first_date_final)
col2.metric("Negative Earnings Period: ", f"{negative_earnings_period} Days")
col3.metric("Money to burn: ", f"${total_spend_to_profit:,.0f}")
col4.metric("Total User Acquisition: ", f"{total_spend_to_profit/cpi:,.0f}")

# chart
fig, ax = plt.subplots(figsize = (25,10))
plt.style.use('ggplot') 

plt.title ("Spend, revenue simulation")
sns.barplot(x = 'Date',
            y = 'values',
            hue = 'type_group',
            data = df_to_plot[df_to_plot.type_group == "renew"],
            ax = ax)

ax.axhline(
    y=daily_spend, 
    color='red',  
    linestyle='dashed',  
    linewidth=2  
)

plt.xticks(rotation=90)
plt.xlabel("Date")
plt.ylabel("Revenue - Spend ($)")
plt.tight_layout()



tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
tab1.subheader("Spend, revenue simulation")
tab1.pyplot(fig)
tab2.subheader("Data simulation")
tab2.write(df_to_calculate)


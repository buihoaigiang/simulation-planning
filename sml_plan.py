# import streamlit as st
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import seaborn as sns
# from datetime import date, timedelta
# # Set page infor ....
# st.set_page_config(
#     page_title="Hoai - Ex-stream-ly Cool App",
#     page_icon=":ice_cube:",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )
# #background color
# page_bg_color = "#e0e0e0"
# page_bg_img = f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
#     background-color: {page_bg_color};
#     font-family: sans-serif;
# }}
# </style>
# """
# st.markdown(page_bg_img, unsafe_allow_html=True)
# # Set sidebar input....
# #input date range, spend, cpi...
# with st.sidebar:
#     start_date = st.date_input('Enter the FROM DATE: ', date.today() + timedelta(days=1))
#     end_date = st.date_input('Enter the TO DATE: ', date.today() + timedelta(days=120))
#     daily_spend = st.number_input('Enter the daily SPEND ($): ', value= 15000, step=1000)
#     cpi = st.number_input('Enter the daily CPI ($): ',  value= 1.8, step=0.2)
#     price = st.number_input('Enter the daily PRICE ($): ', value= 6.99, step= 0.5)
#     st.selectbox('Select SUBSCRIPTION TYPE', ["WEEKLY", "YEARLY"])
#     First_conversion = st.number_input('Enter first conversion rate: ', value= 0.08, step= 0.01)
#     st.markdown("<h1 style='text-align: left; font-size: 16px;'>Enter Retention Rates: </h1>", unsafe_allow_html=True)
# #input rr
#     def calculate_rate(df):
#         df['Rate*'] = df['ASC.Rate'] * First_conversion
#         df['Rate*'] = df['Rate*'].round(3)
#         return df

#     def update_df(df):
#         st.session_state.edited_df = calculate_rate(df)
        
#     dfrr = pd.DataFrame(
#         [
#             {"Period": "First conv.",  "ASC.Rate": First_conversion / First_conversion},
#             {"Period": "1",  "ASC.Rate": 0.60},
#             {"Period": "2",  "ASC.Rate": 0.55},
#             {"Period": "3",  "ASC.Rate": 0.50},
#             {"Period": "4",  "ASC.Rate": 0.45},
#             {"Period": "5",  "ASC.Rate": 0.40},
#             {"Period": "6",  "ASC.Rate": 0.30},
#             {"Period": "7",  "ASC.Rate": 0.25},
#             {"Period": "8",  "ASC.Rate": 0.20},
#             {"Period": "9",  "ASC.Rate": 0.15},
#             {"Period": "10", "ASC.Rate": 0.15},
#             {"Period": "11", "ASC.Rate": 0.15}
#         ]
#     )

#     dfrr = calculate_rate(dfrr)

#     if 'edited_df' not in st.session_state:
#         st.session_state.edited_df = dfrr.copy()

#     edited_df = st.data_editor(st.session_state.edited_df, num_rows="dynamic")

#     if st.button("Calculate"):
#         update_df(edited_df)
        
# #Calculate install
# def define_date(start_date, end_date, daily_spend, cpi, price):
#     date_range = pd.date_range(start=start_date, end=end_date)
#     df = pd.DataFrame(date_range, columns=['Date'])
#     df["install_date"] = df["Date"]
#     df["spend"] = daily_spend
#     df["cpi"] = cpi
#     df["num_install"] = df["spend"] / df["cpi"]
#     df["price"] = price
#     return df
# #Process rr
# def define_renew(df, *retention_rates):
#     for i, rr in enumerate(retention_rates, start=1):
#         df[f"rr{i}"] = rr
#         df[f"remaining_user_ after_rr{i}"] = rr * df["num_install"]
#         renew_offset = 3 if i == 1 else 3 + 7 * (i - 1)
#         df[f"renew_{i}_at"] = df["install_date"] + pd.Timedelta(days=renew_offset)
# retention_rates = edited_df.iloc[:, 2].tolist()
# df = define_date(start_date, end_date, daily_spend, cpi, price)
# define_renew(df, *retention_rates)
# #Creat data frame
# def create_combined_dataframe(df):
#   columns = ["Date", "install_date", "type", "num", "values"]
#   df_install = pd.DataFrame(columns=columns)
#   df_spend = pd.DataFrame(columns=columns)
#   df_install["Date"] = df["Date"]
#   df_install["install_date"] = df["install_date"]
#   df_install["type"] = "install"
#   df_install["num"] = df["num_install"]
#   df_install["values"] = 0
#   df_spend["Date"] = df["Date"]
#   df_spend["install_date"] = df["install_date"]
#   df_spend["type"] = "spend"
#   df_spend["num"] = 0
#   df_spend["values"] = df["spend"]
#   # Combine
#   df_to_calculate = pd.concat([df_install, df_spend], ignore_index=True)
#   # Renew and remaining user
#   renew_at_columns = [col for col in df.columns if col.startswith("renew_") and col.endswith("_at")]
#   remaining_user_columns = [col for col in df.columns if col.startswith("remaining_user_")]
#   # Process renewals
#   for renew_col, remaining_col in zip(renew_at_columns, remaining_user_columns):
#     df_renew = pd.DataFrame(columns=columns)
#     df_renew["Date"] = df[renew_col]
#     df_renew["install_date"] = df["install_date"]
#     type_suffix = renew_col.split("_")[1]
#     df_renew["type"] = f"renew{type_suffix}"
#     df_renew["num"] = df[remaining_col]
#     df_renew["values"] = df[remaining_col] * price
#     # Append
#     df_to_calculate = pd.concat([df_to_calculate, df_renew], ignore_index=True)
#   return df_to_calculate
# #Process final df for daily viz
# df_to_calculate = create_combined_dataframe(df)
# def remove_digits(text):
#     return ''.join(char for char in text if not char.isdigit())
# df_to_calculate['type_group'] = df_to_calculate['type'].apply(remove_digits)
# df_to_calculate["Date"] =pd.to_datetime(df_to_calculate["Date"])
# df_to_plot = df_to_calculate[df_to_calculate.type_group != "install"]
# df_to_plot = df_to_plot.groupby(["Date","type_group"])["values"].sum().reset_index()

# #Process final df for running total viz
# def calculate_running_total(df):
#   df = df.sort_values(['type_group', 'Date'])  # Ensure correct sorting
#   df['running_total'] = df.groupby('type_group')['values'].cumsum()
#   df['Date'] = pd.to_datetime(df['Date'])
#   return df
# df_running = calculate_running_total(df_to_plot)

# #calculate when daily break - even
# df_filtered = df_to_plot[df_to_plot['type_group'] == 'renew']
# df_merged = df_filtered.merge(df_to_plot[df_to_plot['type_group'] == 'spend'], how='left', on='Date')
# first_date = df_merged[df_merged['values_x'] >= df_merged['values_y']]['Date'].iloc[0]
# first_date_final = first_date.strftime('%Y-%m-%d')

# # print(f"First date where revenue is equal to or greater than spend: {first_date.strftime('%Y-%m-%d')}")
# minus_rev_daily = df_merged[df_merged['values_x'] < df_merged['values_y']]
# negative_earnings_period = len(minus_rev_daily)
# total_spend_to_profit = minus_rev_daily.values_y.sum()

# #calculate when running break event
# df_filtered2 = df_running[df_running['type_group'] == 'renew']
# df_merged2 = df_filtered2.merge(df_running[df_running['type_group'] == 'spend'], how='left', on='Date')
# first_date2 = df_merged2[df_merged2['running_total_x'] >= df_merged2['running_total_y']]['Date'].iloc[0]
# first_date_final2 = first_date2.strftime('%Y-%m-%d')

# minus_rev_running2 = df_merged2[df_merged2['running_total_x'] < df_merged2['running_total_y']]
# negative_earnings_period2 = len(minus_rev_running2)


# max_spend_date = df_merged2.loc[df_merged2['running_total_y'].idxmax(), 'Date']
# # Calculate the difference between running_total_y and running_total_x at max_spend_date
# max_spend_value = df_merged2[df_merged2['Date'] == max_spend_date]['running_total_y'].values[0]
# max_renew_value = df_merged2[df_merged2['Date'] == max_spend_date]['running_total_x'].values[0]
# difference = max_renew_value - max_spend_value


# #metric tab
# # col1, col2, col3, col4, col5, col6 = st.columns(6)
# col4, = st.columns(1) 
# col1, col2, col3 = st.columns(3)
# col5, col6, col7 = st.columns(3)

# col1.metric("DAILY - Break-even(BE): ", first_date_final)
# col2.metric("DAILY - Negative Earnings Period: ", f"{negative_earnings_period} Days")
# col3.metric("DAILY SPEND TO BE: ", f"${total_spend_to_profit:,.0f}")
# col4.metric("Total User Acquisition: ", f"{total_spend_to_profit/cpi:,.0f}")
# col5.metric("RUNNING - Break-even: ", first_date_final2)
# col6.metric("RUNNING - Negative Earnings Period: ", f"{negative_earnings_period2} Days")
# col7.metric(f"Profit at {end_date} : ", f"{difference:,.0f}$")

# st.markdown("### Nếu không ra kết quả thì có nghĩa là trong khoảng thời gian này + với các chỉ số input pj không hòa vốn! :white_frowning_face:")
# # chart
# fig, (ax1, ax2) = plt.subplots(2,1,figsize = (30,15))
# plt.style.use('ggplot')
# plt.title ("Spend, revenue simulation")
# sns.lineplot(x = 'Date',
#             y = 'values',
#             data = df_to_plot[df_to_plot.type_group == "renew"],
#             label='Renewal revenue',
#             ax = ax1)
# ax1.axhline(
#     y=daily_spend,
#     color='r',
#     linestyle='dashed',
#     linewidth=2,
#     label='Daily SPEND ($)'
# )
# target_date = pd.to_datetime(first_date_final)  # Convert string to datetime
# ax1.axvline(
#     x=target_date,
#     color="b",
#     linestyle="dashed",
#     linewidth=1,
#     label="Break-even (daily)"  # Add label for clarity
# )

# ax1.set_xlabel("Date")  # Custom x-label for ax1
# ax1.set_ylabel("Daily Revenue - Spend ($)")  # Custom y-label for ax1
# ax1.legend(loc='upper right')  # Add legend with title
# ax1.set_title('DAILY - Renewal Revenue vs Spend')

# sns.lineplot(x='Date', y='running_total', hue='type_group', data=df_running, ax=ax2)

# for i, row in df_filtered.iterrows():
#     if i % 7 == 0:
#         rounded_value = round(row['values'],0)
#         ax1.annotate(f"{rounded_value:.0f}", (row['Date'], row['values']), xytext=(5, 10),
#                      textcoords='offset points', fontsize=12, arrowprops=dict(facecolor='black', shrink=0.05))
        
# target_date2 = pd.to_datetime(first_date_final2) 
#  # Convert string to datetime
# ax2.axvline(
#     x=target_date2,
#     color="b",
#     linestyle="dashed",
#     linewidth=1,
#     label="Break-even (daily)")  # Add label for clarity
# # old_handles, labels = ax.get_legend_handles_labels()
# # plt.legend(handles=old_handles)
# ax2.set_xlabel("Date")  # Custom x-label for ax1
# ax2.set_ylabel("Running Revenue - Spend ($)")  # Custom y-label for ax1
# ax2.legend(loc='upper right')
# ax2.set_title('CUMULATIVE: Renewal Revenue vs Spend')

# plt.tight_layout()
# #make tab
# tab1, tab2 = st.tabs([":chart_with_upwards_trend: Chart", ":card_file_box: Data"])
# tab1.subheader("Spend, revenue simulation")
# tab1.pyplot(fig)
# tab2.subheader("Data simulation")
# tab2.write(df_to_calculate)

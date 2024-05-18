
# In[5]:

import numpy as np
import pandas as pd

import seaborn as sns 
import matplotlib.pyplot as plt 
from colorama import Fore

from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from sklearn.preprocessing import LabelEncoder
import warnings # Supress warnings 
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
np.random.seed(7)


# 数据导入
df = pd.read_csv(r"C:\Users\qing\Desktop\全部数据\附件2-行业日负荷数据.csv")
weather = pd.read_csv(r"C:\Users\qing\Desktop\全部数据\附件3-气象数据.csv")
metrics_df=pd.pivot_table(df,values='有功功率最大值（kw）',index='数据时间',columns='行业类型')
metrics_df.head()

metrics_df1=pd.pivot_table(df,values='有功功率最小值（kw）',index='数据时间',columns='行业类型')
metrics_df1.head()

metrics_df['数据时间'] = metrics_df.index
metrics_df1['数据时间'] = metrics_df1.index


metrics_df= metrics_df.rename(columns={'数据时间':'date'})
metrics_df1= metrics_df1.rename(columns={'数据时间':'date'})

metrics_df = pd.DataFrame(metrics_df.reset_index())
metrics_df1 = pd.DataFrame(metrics_df1.reset_index())


metrics_df
metrics_df1 


# # 商业最大总总有功功率预测
df = metrics_df.fillna(0)
univariate_df = df[['date', '商业']].copy()
univariate_df.columns = ['ds', 'y']
univariate_df
print(univariate_df)

# also: _hourly, _daily, _weekly, or _yearly
from autots import AutoTS

# 模型实例化
model = AutoTS(
    forecast_length=91,
    no_negatives=True,
    frequency='infer',
    ensemble='simple',
    max_generations=10,
    num_validations=2,
    prediction_interval=0.9,
    validation_method="backwards"
)
# 训练模型
model = model.fit(univariate_df, date_col='ds', value_col='y')
# Print the name of the best model
print(model)

prediction = model.predict()
# plot a sample
prediction.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0],
                start_date="2021-01-01")
# 打印出最好的模型的细节
print(model)

# 点预测dataframe
forecasts_df = prediction.forecast
# 上、下限预测
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

# 所有试验模型结果的准确性
model_results = model.results()
# 交叉验证的结果
validation_results = model.results("validation")


# # 查看预测结果
forecasts_df.plot()

forecasts_df
forecasts_df.to_csv('C:\/Users\qing\Desktop\全部数据/商业最大总总有功功率预测结果.csv')
print("商业用电最大总有功功率预测完成")

# # 商业最小总有功功率预测
df = metrics_df1.fillna(0)
univariate_df = df[['date', '商业']].copy()
univariate_df.columns = ['ds', 'y']
# also: _hourly, _daily, _weekly, or _yearly
from autots import AutoTS
# 模型实例化
model = AutoTS(
    forecast_length=91,
    no_negatives=True,
    frequency='infer',
    ensemble='simple',
    max_generations=10,
    num_validations=2,
    prediction_interval=0.9,
    validation_method="backwards"
)
# 训练模型
model = model.fit(univariate_df, date_col='ds', value_col='y')
# Print the name of the best model
print(model)
prediction = model.predict()
# plot a sample
prediction.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0],
                start_date="2021-01-01")
# 打印出最好的模型的细节
print(model)

# 点预测dataframe
forecasts_df = prediction.forecast
# 上、下限预测
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

# 所有试验模型结果的准确性
model_results = model.results()
# 交叉验证的结果
validation_results = model.results("validation")

forecasts_df.plot()

forecasts_df.to_csv('C:\/Users\qing\Desktop\全部数据/商业最小总有功功率预测结果.csv')
print("商业用电最小总有功功率预测完成")

# 大工业用电最大总有功功率预测 
df = metrics_df.fillna(0)
univariate_df = df[['date', '大工业用电']].copy()
univariate_df.columns = ['ds', 'y']
from autots import AutoTS
# 模型实例化
model = AutoTS(
    forecast_length=91,
    no_negatives=True,
    frequency='infer',
    ensemble='simple',
    max_generations=10,
    num_validations=2,
    prediction_interval=0.9,
    validation_method="backwards"
)
# 训练模型
model = model.fit(univariate_df, date_col='ds', value_col='y')
# Print the name of the best model
print(model)

prediction = model.predict()
# plot a sample
prediction.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0],
                start_date="2021-01-01")
# 打印出最好的模型的细节
print(model)

# 点预测dataframe
forecasts_df = prediction.forecast
# 上、下限预测
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

# 所有试验模型结果的准确性
model_results = model.results()
# 交叉验证的结果
validation_results = model.results("validation")
forecasts_df.plot()
forecasts_df.to_csv('C:\/Users\qing\Desktop\全部数据/大工业用电最大总有功功率预测结果.csv')
print("大工业用电最大总有功功率预测完成")


#大工业用电最小总有功功率预测
df = metrics_df1.fillna(0)
univariate_df = df[['date', '大工业用电']].copy()
univariate_df.columns = ['ds', 'y']
model = AutoTS(
    forecast_length=91,
    no_negatives=True,
    frequency='infer',
    ensemble='simple',
    max_generations=3,
    num_validations=1,
)
# 训练模型
model = model.fit(univariate_df, date_col='ds', value_col='y')
# Print the name of the best model
print(model)
prediction = model.predict()
# plot a sample
prediction.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0],
                start_date="2021-01-01")
# 打印出最好的模型的细节
print(model)

# 点预测dataframe
forecasts_df = prediction.forecast
# 上、下限预测
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

# 所有试验模型结果的准确性
model_results = model.results()
# 交叉验证的结果
validation_results = model.results("validation")
forecasts_df.plot()
forecasts_df.to_csv('C:\/Users\qing\Desktop\全部数据/大工业用电最小总有功功率预测结果.csv')
print("大工业用电最小总有功功率预测完成")


#普通工业最大总有功功率预测
df = metrics_df.fillna(0)
univariate_df = df[['date', '普通工业']].copy()
univariate_df.columns = ['ds', 'y']
model = AutoTS(
    forecast_length=91,
    frequency='infer',
    ensemble='simple',
    max_generations=4,
    num_validations=2,
    prediction_interval=0.9,
)
# 训练模型
model = model.fit(univariate_df, date_col='ds', value_col='y')
# Print the name of the best model
print(model)
prediction = model.predict()
# plot a sample
prediction.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0],
                start_date="2021-01-01")
# 打印出最好的模型的细节
print(model)

# 点预测dataframe
forecasts_df = prediction.forecast
# 上、下限预测
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

# 所有试验模型结果的准确性
model_results = model.results()
# 交叉验证的结果
validation_results = model.results("validation")
forecasts_df.plot()

forecasts_df.to_csv('C:\/Users\qing\Desktop\全部数据/普通工业最大总有功功率预测结果.csv')
print("普通工业用电最大总有功功率预测完成")


# 普通工业最小总有功功率预测 
df = metrics_df1.fillna(0)
univariate_df = df[['date', '普通工业']].copy()
univariate_df.columns = ['ds', 'y']

model = AutoTS(
    forecast_length=91,
    frequency='infer',
    ensemble='simple',
    max_generations=4,
    num_validations=2,
    prediction_interval=0.9,
)
# 训练模型
model = model.fit(univariate_df, date_col='ds', value_col='y')
# Print the name of the best model
print(model)
prediction = model.predict()
# plot a sample
prediction.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0],
                start_date="2021-01-01")
# 打印出最好的模型的细节
print(model)

# 点预测dataframe
forecasts_df = prediction.forecast
# 上、下限预测
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

# 所有试验模型结果的准确性
model_results = model.results()
# 交叉验证的结果
validation_results = model.results("validation")
forecasts_df.plot()

forecasts_df.to_csv('C:\/Users\qing\Desktop\全部数据/普通工业最小总有功功率预测结果.csv')
print("普通工业用电最小总有功功率预测完成")


# 非普通工业最大总有功功率预测
df = metrics_df.fillna(0)
univariate_df = df[['date', '非普工业']].copy()
univariate_df.columns = ['ds', 'y']
univariate_df=univariate_df.tail(691)

model = AutoTS(
    forecast_length=91,
    frequency='infer',
    ensemble='simple',
    max_generations=4,
    num_validations=2,
    prediction_interval=0.9,
)
# 训练模型
model = model.fit(univariate_df, date_col='ds', value_col='y')
# Print the name of the best model
print(model)
prediction = model.predict()
# plot a sample
prediction.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0],
                start_date="2021-01-01")
# 打印出最好的模型的细节
print(model)

# 点预测dataframe
forecasts_df = prediction.forecast
# 上、下限预测
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

# 所有试验模型结果的准确性
model_results = model.results()
# 交叉验证的结果
validation_results = model.results("validation")
forecasts_df.plot()
forecasts_df.to_csv('C:\/Users\qing\Desktop\全部数据/非普通工业最大总有功功率预测结果.csv')
print("非普通工业用电最大总有功功率预测完成")

# 普通工业最小总有功功率预测
df = metrics_df1.fillna(0)
univariate_df = df[['date', '非普工业']].copy()
univariate_df.columns = ['ds', 'y']
model = AutoTS(
    forecast_length=91,
    frequency='infer',
    ensemble='simple',
    max_generations=4,
    num_validations=2,
    prediction_interval=0.9,
)
# 训练模型
model = model.fit(univariate_df, date_col='ds', value_col='y')
# Print the name of the best model
print(model)
prediction = model.predict()
# plot a sample
prediction.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0],
                start_date="2021-01-01")
# 打印出最好的模型的细节
print(model)

# 点预测dataframe
forecasts_df = prediction.forecast
# 上、下限预测
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

# 所有试验模型结果的准确性
model_results = model.results()
# 交叉验证的结果
validation_results = model.results("validation")
forecasts_df.plot()
forecasts_df.to_csv('C:\/Users\qing\Desktop\全部数据/非普通工业最小总有功功率预测结果.csv')
print("非普通工业用电最小总有功功率预测完成")
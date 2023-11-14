import math  # 导入数学库，用于数学计算
import pandas as pd  # 导入Pandas库，用于数据处理
import sklearn.metrics as metrics  # 导入Scikit-learn库中的metrics模块，用于性能评估
from lightgbm import LGBMRegressor  # 导入LightGBM库中的LGBMRegressor模型
from sklearn.metrics import r2_score  # 导入Scikit-learn库中的r2_score模块，用于计算决定系数
from sklearn.model_selection import train_test_split  # 导入Scikit-learn库中的train_test_split函数，用于划分训练集和测试集
from xgboost import XGBRegressor  # 导入XGBoost库中的XGBRegressor模型

sample_submission = pd.read_csv(
    "C:/Users\zmy\PycharmProjects\pydl\data\House Prices - Advanced Regression Techniques/sample_submission.csv")  # 从CSV文件中读取sample_submission数据集
test = pd.read_csv(
    "C:/Users\zmy\PycharmProjects\pydl\data\House Prices - Advanced Regression Techniques/test.csv")  # 从CSV文件中读取test数据集
train = pd.read_csv(
    "C:/Users\zmy\PycharmProjects\pydl\data\House Prices - Advanced Regression Techniques/train.csv")  # 从CSV文件中读取train数据集

c_test = test.copy()  # 创建test数据集的副本
c_train = train.copy()  # 创建train数据集的副本
# 在c_train中添加名为'train'的列，所有值为1
c_train['train'] = 1
# 在c_test中添加名为'train'的列，所有值为0
c_test['train'] = 0
# 将c_train和c_test沿着行的方向进行合并，生成新的数据框df
df = pd.concat([c_train, c_test], axis=0, sort=False)

# 计算每列缺失值的百分比
NAN = [(c, df[c].isna().mean() * 100) for c in df]
# 将缺失值百分比数据转换为DataFrame
NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])
# 保留缺失值百分比大于50%的列
NAN = NAN[NAN.percentage > 50]
# 按百分比降序排序
NAN.sort_values("percentage", ascending=False)

# 删除指定的列，这些列包含了'Alley','PoolQC','Fence','MiscFeature'
df = df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

# 选择数据框中类型为对象（字符串等非数值型）的列
object_columns_df = df.select_dtypes(include=['object'])
# 选择数据框中类型不为对象（数值型）的列
numerical_columns_df = df.select_dtypes(exclude=['object'])

# 统计对象类型的列中每列的缺失值数量
null_counts = object_columns_df.isnull().sum()
# 打印每列的缺失值数量
print("Number of null values in each column:\n{}".format(null_counts))

# 定义需要填充为'None'的列列表
columns_None = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish',
                'GarageQual', 'FireplaceQu', 'GarageCond']
# 使用'None'填充指定列
object_columns_df.loc[:, columns_None] = object_columns_df.loc[:, columns_None].fillna('None')
# 定义需要用众数填充的列列表
columns_with_lowNA = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Electrical', 'KitchenQual',
                      'Functional', 'SaleType']
# 使用众数填充指定列
object_columns_df[columns_with_lowNA] = object_columns_df[columns_with_lowNA].fillna(object_columns_df.mode().iloc[0])

# 计算数值列中每列的缺失值数量
null_counts = numerical_columns_df.isnull().sum()
# 打印每列的缺失值数量
print("Number of null values in each column:\n{}".format(null_counts))

# 打印年份差值的中位数，表示房屋出售年份与建造年份之间的差异
print((numerical_columns_df['YrSold'] - numerical_columns_df['YearBuilt']).median())
# 打印LotFrontage列的中位数
print(numerical_columns_df["LotFrontage"].median())

# 使用YrSold减去35填充GarageYrBlt列的缺失值
numerical_columns_df['GarageYrBlt'] = numerical_columns_df['GarageYrBlt'].fillna(
    numerical_columns_df['YrSold'] - 35)
# 使用68填充LotFrontage列的缺失值
numerical_columns_df.loc[:, 'LotFrontage'] = numerical_columns_df['LotFrontage'].fillna(68)
# 使用0填充所有数值类型列中的缺失值
numerical_columns_df = numerical_columns_df.fillna(0)

# 绘制不同列的条形图，显示不同取值的数量分布
# 打印列中每个取值的数量
object_columns_df['Utilities'].value_counts().plot(kind='bar', figsize=[10, 3])
object_columns_df['Utilities'].value_counts()
object_columns_df['Street'].value_counts().plot(kind='bar', figsize=[10, 3])
object_columns_df['Street'].value_counts()
object_columns_df['Condition2'].value_counts().plot(kind='bar', figsize=[10, 3])
object_columns_df['Condition2'].value_counts()
object_columns_df['RoofMatl'].value_counts().plot(kind='bar', figsize=[10, 3])
object_columns_df['RoofMatl'].value_counts()
object_columns_df['Heating'].value_counts().plot(kind='bar', figsize=[10, 3])
object_columns_df['Heating'].value_counts()
# 从DataFrame中删除指定的列
object_columns_df = object_columns_df.drop(['Heating', 'RoofMatl', 'Condition2', 'Street', 'Utilities'], axis=1)

# 计算房屋的年龄
numerical_columns_df['Age_House'] = (numerical_columns_df['YrSold'] - numerical_columns_df['YearBuilt'])
# 查看描述统计信息
numerical_columns_df['Age_House'].describe()
# 找到年龄为负值的房屋信息
Negatif = numerical_columns_df[numerical_columns_df['Age_House'] < 0]

# 将售出年份早于建造年份的记录中的售出年份设为2009
numerical_columns_df.loc[numerical_columns_df['YrSold'] < numerical_columns_df['YearBuilt'], 'YrSold'] = 2009
# 计算更新后的房屋年龄
numerical_columns_df['Age_House'] = (numerical_columns_df['YrSold'] - numerical_columns_df['YearBuilt'])
# 描述房屋年龄的统计信息
numerical_columns_df['Age_House'].describe()

# 计算总地下室浴室数，将半浴室按照0.5计算
numerical_columns_df['TotalBsmtBath'] = numerical_columns_df['BsmtFullBath'] + numerical_columns_df[
    'BsmtFullBath'] * 0.5
# 计算总浴室数，将半浴室按照0.5计算
numerical_columns_df['TotalBath'] = numerical_columns_df['FullBath'] + numerical_columns_df['HalfBath'] * 0.5
# 计算总面积，包括地下室、一楼和二楼
numerical_columns_df['TotalSA'] = numerical_columns_df['TotalBsmtSF'] + numerical_columns_df['1stFlrSF'] + \
                                  numerical_columns_df['2ndFlrSF']

# 对分类特征进行编码，有序分类特征需要进行从0到N的映射
bin_map = {'TA': 2, 'Gd': 3, 'Fa': 1, 'Ex': 4, 'Po': 1, 'None': 0, 'Y': 1, 'N': 0, 'Reg': 3, 'IR1': 2, 'IR2': 1,
           'IR3': 0, "None": 0,
           "No": 2, "Mn": 2, "Av": 3, "Gd": 4, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6
           }
object_columns_df['ExterQual'] = object_columns_df['ExterQual'].map(bin_map)
object_columns_df['ExterCond'] = object_columns_df['ExterCond'].map(bin_map)
object_columns_df['BsmtCond'] = object_columns_df['BsmtCond'].map(bin_map)
object_columns_df['BsmtQual'] = object_columns_df['BsmtQual'].map(bin_map)
object_columns_df['HeatingQC'] = object_columns_df['HeatingQC'].map(bin_map)
object_columns_df['KitchenQual'] = object_columns_df['KitchenQual'].map(bin_map)
object_columns_df['FireplaceQu'] = object_columns_df['FireplaceQu'].map(bin_map)
object_columns_df['GarageQual'] = object_columns_df['GarageQual'].map(bin_map)
object_columns_df['GarageCond'] = object_columns_df['GarageCond'].map(bin_map)
object_columns_df['CentralAir'] = object_columns_df['CentralAir'].map(bin_map)
object_columns_df['LotShape'] = object_columns_df['LotShape'].map(bin_map)
object_columns_df['BsmtExposure'] = object_columns_df['BsmtExposure'].map(bin_map)
object_columns_df['BsmtFinType1'] = object_columns_df['BsmtFinType1'].map(bin_map)
object_columns_df['BsmtFinType2'] = object_columns_df['BsmtFinType2'].map(bin_map)
# 映射PavedDrive列的值为数字
PavedDrive = {"N": 0, "P": 1, "Y": 2}
object_columns_df['PavedDrive'] = object_columns_df['PavedDrive'].map(PavedDrive)

# 选择 object 数据类型的列并将其存储在 rest_object_columns 中
rest_object_columns = object_columns_df.select_dtypes(include=['object'])
# 应用独热编码将分类变量转换为数值表示
object_columns_df = pd.get_dummies(object_columns_df, columns=rest_object_columns.columns)
# 将经过独热编码处理的分类变量和数值变量拼接成最终的数据集 df_final
df_final = pd.concat([object_columns_df, numerical_columns_df], axis=1, sort=False)

# 从最终数据集 df_final 中删除不需要的列 'Id'
df_final = df_final.drop(['Id', ], axis=1)
# 划分训练集 df_train，其中 'train' 列为 1 表示训练集样本
df_train = df_final[df_final['train'] == 1]
df_train = df_train.drop(['train', ], axis=1)
# 划分测试集 df_test，同时删除 'SalePrice' 列和 'train' 列
df_test = df_final[df_final['train'] == 0]
df_test = df_test.drop(['SalePrice'], axis=1)
df_test = df_test.drop(['train', ], axis=1)
# 提取目标变量 'SalePrice' 到 target
target = df_train['SalePrice']
# 从 df_train 中删除 'SalePrice' 列，作为模型的训练集特征
df_train = df_train.drop(['SalePrice'], axis=1)

# 划分训练集和测试集，测试集：训练集=1：2
x_train, x_test, y_train, y_test = train_test_split(df_train, target, test_size=0.33, random_state=0)

# 初始化XGBoost回归模型
xgb = XGBRegressor(booster='gbtree', colsample_bylevel=1,
                   colsample_bynode=1, colsample_bytree=0.6, gamma=0,
                   importance_type='gain', learning_rate=0.01, max_delta_step=0,
                   max_depth=4, min_child_weight=1.5, n_estimators=2400,
                   n_jobs=1, nthread=None, objective='reg:squarederror',
                   reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1,
                   silent=None, subsample=0.8, verbosity=1)

# 初始化LightGBM回归模型
lgbm = LGBMRegressor(objective='regression',
                     num_leaves=4,
                     learning_rate=0.01,
                     n_estimators=12000,
                     max_bin=200,
                     colsample_bytree=0.6,
                     subsample=0.8,
                     bagging_fraction=0.75,
                     bagging_freq=5,
                     bagging_seed=7,
                     feature_fraction=0.4)


# 使用训练集训练XGBoost模型和LightGBM模型
xgb.fit(x_train, y_train)
lgbm.fit(x_train, y_train, eval_metric='rmse')
# 使用训练好的模型对测试集进行预测
predict1 = xgb.predict(x_test)
predict = lgbm.predict(x_test)

# 计算并打印XGBoost模型在测试集上的均方根误差
print('Root Mean Square Error test (XGBoost) = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict1))))
# 计算并打印LightGBM模型在测试集上的均方根误差
print('Root Mean Square Error test (LightGBM) = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict))))

# 在整个训练集上拟合XGBoost模型
xgb.fit(df_train, target)
# 在整个训练集上拟合LightGBM模型
lgbm.fit(df_train, target, eval_metric='rmse')
# 预测测试集的房价
predict4 = lgbm.predict(df_test)
predict3 = xgb.predict(df_test)
# 组合两个模型的预测结果，其中XGBoost权重为0.45，LightGBM权重为0.55
predict_y = (predict3 * 0.45 + predict4 * 0.55)

# 计算并打印XGBoost模型在测试集上的决定系数
r2_xgb = r2_score(y_test, predict1)
print('R-squared test (XGBoost) = ' + str(r2_xgb))
# 计算并打印LightGBM模型在测试集上的决定系数
r2_lgbm = r2_score(y_test, predict)
print('R-squared test (LightGBM) = ' + str(r2_lgbm))

# # 创建提交文件
# submission = pd.DataFrame({
#         "Id": test["Id"],
#         "SalePrice": predict_y
#     })
#
# # 将结果保存为CSV文件
# submission.to_csv('submission.csv', index=False)

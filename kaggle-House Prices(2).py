import math
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# 读取数据
train = pd.read_csv("C:/Users/zmy/PycharmProjects/pydl/data/House Prices - Advanced Regression Techniques/train.csv")
test = pd.read_csv("C:/Users/zmy/PycharmProjects/pydl/data/House Prices - Advanced Regression Techniques/test.csv")
sample_submission = pd.read_csv("C:/Users/zmy/PycharmProjects/pydl/data/House Prices - Advanced Regression Techniques/sample_submission.csv")

# 数据处理
train['train'] = 1
test['train'] = 0
df = pd.concat([train, test], axis=0, sort=False)

# 处理缺失值
df = df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])
df['Functional'] = df['Functional'].fillna('Typ')
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# 特征工程
df['TotalBsmtBath'] = df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5
df['TotalBath'] = df['FullBath'] + df['HalfBath'] * 0.5
df['TotalSA'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['Age_House'] = df['YrSold'] - df['YearBuilt']
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YrSold'] - 35)
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# 类别特征编码
ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtCond', 'BsmtQual', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
                'GarageQual', 'GarageCond']
ordinal_mapping = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
for col in ordinal_cols:
    df[col] = df[col].map(ordinal_mapping).fillna(0)

# 独热编码
df = pd.get_dummies(df, drop_first=True)

# 划分数据集
df_train = df[df['train'] == 1].drop(['train'], axis=1)
df_test = df[df['train'] == 0].drop(['SalePrice', 'train'], axis=1)

X = df_train.drop(['Id', 'SalePrice'], axis=1)
y = df_train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# 模型训练
xgb = XGBRegressor(learning_rate=0.01, n_estimators=2400, max_depth=4, min_child_weight=1.5,
                   colsample_bytree=0.6, subsample=0.8, reg_alpha=0.6, reg_lambda=0.6)
lgbm = LGBMRegressor(learning_rate=0.01, n_estimators=12000, max_bin=200, num_leaves=4,
                     colsample_bytree=0.6, subsample=0.8, bagging_fraction=0.75, bagging_freq=5, bagging_seed=7,
                     feature_fraction=0.4)

xgb.fit(X_train, y_train)
lgbm.fit(X_train, y_train, eval_metric='rmse')

# 模型预测
predict_xgb = xgb.predict(X_test)
predict_lgbm = lgbm.predict(X_test)

# 评估模型
rmse_xgb = math.sqrt(mean_squared_error(y_test, predict_xgb))
rmse_lgbm = math.sqrt(mean_squared_error(y_test, predict_lgbm))
r2_xgb = r2_score(y_test, predict_xgb)
r2_lgbm = r2_score(y_test, predict_lgbm)

print(f'RMSE (XGBoost): {rmse_xgb}')
print(f'RMSE (LightGBM): {rmse_lgbm}')
print(f'R-squared (XGBoost): {r2_xgb}')
print(f'R-squared (LightGBM): {r2_lgbm}')

# 集成模型预测
predict_y = 0.45 * xgb.predict(df_test.drop(['Id'], axis=1)) + 0.55 * lgbm.predict(df_test.drop(['Id'], axis=1))

# # 保存结果
# submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': predict_y})
# submission.to_csv('submission.csv', index=False)

import pandas as pd

# 读取原始CSV文件
df = pd.read_csv('test.csv')

df.columns = ["zfygpa",
        "zgpa",
        "DOB_yr",
        "weighted_lsat_ugpa",
        "cluster_tier",
        "family_income",
        "lsat",
        "ugpa",
        "isPartTime",
        "sex",
        "race",
        "income"]

df['race-sex'] = df['race'] + '-' + df['sex']

# 独热编码
columns_to_encode = ['isPartTime']
df_encoded = pd.get_dummies(df, columns=columns_to_encode)

# 保存新的CSV文件
df_encoded.to_csv('output_test.csv', index=False)

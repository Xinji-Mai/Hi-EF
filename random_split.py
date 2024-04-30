# 导入pandas库，用于处理csv文件
import pandas as pd
# 导入sklearn库的train_test_split函数，用于随机划分数据集
from sklearn.model_selection import train_test_split

# 读取csv文件，假设文件名为data.csv
data = pd.read_csv("/home/et23-maixj/mxj/DFER_Datasets/MAFW/preprocess/meta.csv")
# 随机划分数据集，指定训练集占比为0.8，测试集占比为0.2
train, test = train_test_split(data, train_size=0.8, test_size=0.2, random_state=42)
# 将训练集和测试集保存为csv文件，命名为train.csv和test.csv
train.to_csv("/home/et23-maixj/mxj/DFER_Datasets/MAFW/preprocess/train_random.csv", index=False)
test.to_csv("/home/et23-maixj/mxj/DFER_Datasets/MAFW/preprocess/test_random.csv", index=False)

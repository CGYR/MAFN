import numpy as np
import pandas as pd
from typing import List, Dict
import re
import json
import scipy.stats as stats


# 长尾特征
LONGTAIL_FEATURES = []


# 缺失值填充字典
FILL_DICT = {
}


class ML_DataPreprocessor():
    """数据预处理模块"""
    def __init__(self,
                data: pd.DataFrame,                                    # 原始数据集
                target_cols: List[str] = ['label', 'label_pay'],        # 标签列表
                fill_dict: dict = {},                                   # 空值填充
                vocab_dict: dict = {},                                  # 离散特征字典
                cutoff_dict: dict = {},
                ):
        super().__init__()
        self.data = data
        self.fill_dict = fill_dict
        # self.target_cols = target_cols
        self.feature_cols = [col for col in data.columns if col not in target_cols ]
        self.cutoff_dict = cutoff_dict
        # self.vocab_dict = vocab_dict
        self.LONGTAIL_FEATURES = []
        
        # # 预处理
        # self.data_preproess()


    def abnormal_preproess(self):
        """异常处理、空值处理"""
        for feature in self.feature_cols:
            if re.search('install',str(feature)):
                self.fill_dict[feature] = 0.0
            if re.search('uninstall',str(feature)):
                self.fill_dict[feature] = 0.0
            if re.search('act_',str(feature)):
                self.fill_dict[feature] = 0.0
            if re.search('pay',str(feature)):
                self.fill_dict[feature] = 0.0
            if re.search('last',str(feature)):
                self.fill_dict[feature] = 0.0


    def custom_binning(self, x, split, bin_uppers):
        """分箱
        中位数split之后，以间隔2做等距分箱
        """
        if x <= split:
            return x
        if x > bin_uppers[-1]:
            return split + len(bin_uppers) + 1   
        for i in range(len(bin_uppers)):
            if x <= bin_uppers[i]:
                return split + i + 1

    '''
    def replace_outliers(df):
        # 计算每个特征的中位数和标准差
        medians = df.median()
        stds = df.std()

        # 查找任何与中位数差距超过3倍标准差的值
        is_outlier = np.abs(df - medians) > (3 * stds)

        # 将异常值替换为0
        df[is_outlier] = 0

        return df
    '''


    def get_long_tial_features(self):
        long_tial_features = []

        # 遍历数据集中的每个特征
        for column in self.feature_cols:
            # 如果特征是数值型数据，则进行长尾分布分析
            if self.data[column].dtype == "float64" or self.data[column].dtype == "int64":
                # 计算偏度和峰度
                skewness = stats.skew(self.data[column])
                kurtosis = stats.kurtosis(self.data[column])

                # 判断特征是否具有长尾分布
                if abs(skewness) > 3: # and kurtosis < 3:
                    long_tial_features.append(column)
        return long_tial_features

    
    def get_long_tial_re_features(self):
        """异常处理、空值处理"""
        for feature in self.feature_cols:
            if re.search('duration',str(feature)):
                self.LONGTAIL_FEATURES.append(str(feature))
            if re.search('total',str(feature)):
                self.LONGTAIL_FEATURES.append(str(feature))


    def data_preproess(self):
        """所有数据预处理：空值填充 + 异常值处理 + 分箱"""
    
        # 合并类别特征
        rom_version = list(self.data['rom_version'].unique())
        os_version = list(self.data['os_version'].unique())
        rom_map = {}
        for item in rom_version:
            item = str(item)
            if '_' in item:
                rom_map[item]=item.split('_')[0]
            elif '_' not in item:
                rom_map[item]=item
                
        os_version_map = {}
        for item in os_version:
            item = str(item)
            if '_' in item:
                os_version_map[item]=item.split('_')[0]
            elif '_' not in item:
                os_version_map[item]=item

        self.data['os_version'] = self.data['os_version'].replace(os_version_map)
        self.data['rom_version'] = self.data['rom_version'].replace(rom_map)


        # 异常值处理
        # 空值填充 + 异常值处理
        print('进行空值填充 + 异常值处理')
        if self.fill_dict == {}:
            self.fill_dict = FILL_DICT
            self.abnormal_preproess()
            for feature in self.feature_cols:  
                if feature in self.fill_dict:
                    self.data[feature] = self.data[feature].replace(-999.0, self.fill_dict[feature])
                    self.data[feature].fillna(self.fill_dict[feature], inplace=True)
        else:
            self.abnormal_preproess()
            for feature in self.feature_cols:
                if feature in self.fill_dict:
                    self.data[feature] = self.data[feature].replace(-999.0, self.fill_dict[feature]) # +1
                    # 验证、测试集截断
                    # self.data.loc[self.data[feature] < self.fill_dict[feature] + 1, feature] = self.fill_dict[feature]
                    self.data[feature].fillna(self.fill_dict[feature], inplace=True)

        # # 分箱
        print('进行分箱')
        self.LONGTAIL_FEATURES = self.get_long_tial_features()
        #self.get_long_tial_re_features()
        print('LONGTAIL_FEATURES: {}'.format(self.LONGTAIL_FEATURES))
        if self.cutoff_dict == {}:
            
            for feature in LONGTAIL_FEATURES:
                num_bins = 50
                self.data[feature] = self.data[feature].fillna(0)
                split = int(self.data[feature].median()) # 中位数
                quantiles = [0.5 + (i + 1) * 0.5 / num_bins for i in range(num_bins)]
                quantiles[-1] = 1.0
                bin_uppers = []
                last_upper = split
                for q in quantiles:
                    upper = self.data[feature].quantile(q)
                    if upper <= last_upper:
                        upper = last_upper + 1
                    last_upper = upper
                    bin_uppers.append(upper)
                self.data[feature] = self.data[feature].apply(lambda x: self.custom_binning(x, split, bin_uppers))
                self.cutoff_dict[feature] = [split, bin_uppers]
                
            tf = open("cutoff_dict.json", "w")
            json.dump(self.cutoff_dict, tf)
            tf.close()
            
        else:
            print('test LONGTAIL_FEATURES')
            for feature in LONGTAIL_FEATURES:
                split, bin_uppers = self.cutoff_dict[feature]
                self.data[feature] = self.data[feature].apply(lambda x: self.custom_binning(x, split, bin_uppers))

        return self.data


if __name__ == '__main__':
    pass
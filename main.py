import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

class HappinessStudy:
    """
    一个用于研究幸福感与婚姻状况关系的类，包含数据处理、有序逻辑回归模型拟合、机器学习模型训练和评估等功能。
    
    该类实现了数据加载、预处理、模型拟合、假设检验、性能评估等完整分析流程。
    """
    
    def __init__(self, data_path='data.xlsx'):
        """
        初始化HappinessStudy对象。
        
        Args:
            data_path (str, optional): 数据文件路径。如果未提供，则使用默认路径。
        """
        self.data = None
        self.data_path = data_path
        self.results = {}
        self.model_results = {}
        self.classification_report = {}
        self.y_pred_proba = {}
        self.scaler = None
        self.smote = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.X_train_resampled = None
        self.y_train_resampled = None
    
    def load_data(self, data_path='data.xlsx'):
        """
        加载数据文件。
        
        Args:
            data_path (str, optional): 数据文件路径。如果未提供，则使用初始化时的路径。
        
        Returns:
            pd.DataFrame: 加载的数据。
        
        Raises:
            FileNotFoundError: 如果数据文件不存在。
            Exception: 如果加载数据时发生其他错误。
        """
        if data_path is None and self.data_path is None:
            raise ValueError("数据路径未提供。请提供数据路径或在初始化时指定。")
        
        if data_path is not None:
            self.data_path = data_path
        
        try:
            self.data = pd.read_excel(self.data_path, sheet_name="data")
            print(f"数据加载成功，形状: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"数据文件 {self.data_path} 不存在。")
        except Exception as e:
            raise Exception(f"加载数据时出错: {str(e)}")

    def preprocess_data(self):
        """
        完整的数据预处理流程，包括异常值处理、缺失值处理、变量编码、对数变换、变量构建等。
        """
        if self.data is None:
            raise ValueError("数据未加载。请先调用load_data方法加载数据。")

        # 1. 异常值处理
        print("开始异常值处理...")
        try:
            numeric_cols = ['income', 'family_income', 'age']
            for col in numeric_cols:
                if col in self.data.columns:
                    self.data = self.data[self.data[col] > 0]

            categorical_cols = ['education', 'socialize', 'equity', 'socio_health_pc']
            for col in categorical_cols:
                if col in self.data.columns:
                    self.data = self.data[self.data[col] != 0]

            print(f"异常值处理完成，剩余样本数: {len(self.data)}")

        except Exception as e:
            print(f"异常值处理失败: {str(e)}")
            return None

        # 2. 缺失值处理
        print("开始缺失值处理...")
        try:
            numeric_columns = ['income', 'family_income', 'age', 'education', 'socialize', 'equity', 'socio_health_pc']
            for col in numeric_columns:
                if col in self.data.columns and self.data[col].isnull().any():
                    median_val = self.data[col].median()
                    self.data[col].fillna(median_val, inplace=True)
                    print(f"{col} 缺失值已用中位数 {median_val:.2f} 填充")

            categorical_columns = ['marital', 'gender', 'class']
            for col in categorical_columns:
                if col in self.data.columns and self.data[col].isnull().any():
                    mode_val = self.data[col].mode()[0]
                    self.data[col].fillna(mode_val, inplace=True)
                    print(f"{col} 缺失值已用众数 {mode_val} 填充")

            print("缺失值处理完成。")

        except Exception as e:
            print(f"缺失值处理失败: {str(e)}")
            return None

        # 3. 变量编码
        print("开始变量编码...")
        try:
            # 教育程度编码
            if 'education' in self.data.columns:
                education_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6, 8: 6}
                self.data['education'] = self.data['education'].map(education_map)

            # 性别编码
            if 'gender' in self.data.columns:
                gender_map = {'女': 0, '男': 1}
                self.data['gender'] = self.data['gender'].map(gender_map)

            # 城乡编码
            if 'urban_rural' in self.data.columns:
                urban_map = {'城市': 1, '农村': 0}
                self.data['urban_rural'] = self.data['urban_rural'].map(urban_map)

            # 社会公平编码
            if 'equity' in self.data.columns:
                equity_map = {'不公平': 1, '一般': 2, '公平': 3}
                self.data['equity'] = self.data['equity'].map(equity_map)

            # 社交频率编码
            if 'socialize' in self.data.columns:
                socialize_map = {'少': 1, '一般': 2, '多': 3}
                self.data['socialize'] = self.data['socialize'].map(socialize_map)

            # 社会阶层编码
            if 'class' in self.data.columns:
                class_map = {'最底层': 1, '底层': 2, '下层': 3, '中下层': 4, '中层': 5, 
                            '中上层': 6, '上层': 7, '顶层': 8, '最顶层': 10}
                self.data['class'] = self.data['class'].map(class_map)

            print("变量编码完成。")

        except Exception as e:
            print(f"变量编码失败: {str(e)}")
            return None

        # 4. 收入对数变换
        print("开始收入对数变换...")
        try:
            if 'income' in self.data.columns:
                self.data['log_income'] = np.log(self.data['income'])
                print("个人收入已进行对数变换")

            if 'family_income' in self.data.columns:
                self.data['log_family_income'] = np.log(self.data['family_income'])
                print("家庭收入已进行对数变换")

            print("收入对数变换完成。")

        except Exception as e:
            print(f"收入对数变换失败: {str(e)}")
            return None

        # 5. 年龄中心化和平方项
        print("开始构建年龄相关变量...")
        try:
            if 'age' in self.data.columns:
                # 年龄中心化（减去均值）
                age_mean = self.data['age'].mean()
                self.data['age_centered'] = self.data['age'] - age_mean
                print(f"年龄中心化完成，均值: {age_mean:.2f}")

                # 年龄平方项
                self.data['age_centered_sq'] = self.data['age_centered'] ** 2
                print("年龄平方项构建完成")

            print("年龄相关变量构建完成。")

        except Exception as e:
            print(f"年龄相关变量构建失败: {str(e)}")
            return None

        # 6. 分组划分
        print("开始分组划分...")
        try:
            # 创建分组标识
            self.data['group'] = self.data['birth'].apply(lambda x: 
                '40-50后' if x <= 1959 else 
                '60-70后' if 1960 <= x <= 1979 else 
                '80-90后')

            # 创建分组数据
            groups = {}
            for group_name in ['40-50后', '60-70后', '80-90后']:
                groups[group_name] = self.data[self.data['group'] == group_name].copy()

            # 从数据中移除'birth'列
            self.data = self.data.drop(columns=['birth', 'group'])

            # 选择最终使用的变量
            final_columns = ['marital', 'survey_type', 'gender', 'education', 'urban_rural',
                            'socialize', 'equity', 'socio_health_pc', 'age_centered', 'age_centered_sq',
                            'log_income', 'log_family_income', 'class', 'happiness']

            # 保留相关变量
            self.data = self.data[final_columns]

            print("分组划分完成。")
            return groups

        except Exception as e:
            print(f"分组划分失败: {str(e)}")
            return None
    
    def fit_ordinal_logit_model(self, group_data=None, group_name="全体样本"):
        """
        拟合有序逻辑回归模型。
        """
        if group_data is None:
            group_data = self.data.copy()

        # 确保所有变量都存在
        required_cols = ['marital', 'survey_type', 'gender', 'education', 'urban_rural',
                        'socialize', 'equity', 'socio_health_pc', 'age_centered', 'age_centered_sq',
                        'log_income', 'log_family_income', 'class']

        missing_cols = [col for col in required_cols if col not in group_data.columns]
        if missing_cols:
            print(f"警告: 缺少以下变量: {missing_cols}")
            return None

        try:
            y = group_data['happiness'].astype(int)
            X = group_data[required_cols]

            mod = OrderedModel(y, X, distr='logit')
            res = mod.fit(method='bfgs', disp=False)

            # 计算伪R方
            null_X = np.zeros((len(y), 1))
            null_mod = OrderedModel(y, null_X, distr='logit')
            null_res = null_mod.fit(method='bfgs', disp=False)
            pseudo_r2 = 1 - (res.llf / null_res.llf)

            results = pd.DataFrame({
                'Coef': res.params,
                'Std.Err': res.bse,
                'P>|z|': res.pvalues,
                '[0.025': res.conf_int()[0],
                '0.975]': res.conf_int()[1],
            }).round(3)

            pseudo_row = pd.DataFrame({'Pseudo R-squared': [pseudo_r2]}, index=['Pseudo R-squared'])
            results = pd.concat([results, pseudo_row])

            self.results[group_name] = results
            print(f"有序逻辑回归模型拟合完成。分组: {group_name} (N={len(group_data)})")
            return results

        except Exception as e:
            print(f"拟合有序逻辑回归模型失败: {str(e)}")
            return None
    def perform_parallelism_test(self, group_data=None, group_name="全体样本"):
        """
        执行平行性假设检验。
        
        Args:
            group_data (pd.DataFrame, optional): 分组数据。如果未提供，则使用全体数据。
            group_name (str, optional): 分组名称。
        
        Returns:
            tuple: 似然比统计量和p值。
        """
        if group_data is None:
            group_data = self.data.copy()
        
        y = group_data['happiness'].astype(int)
        X = group_data[['marital', 'survey_type', 'gender', 'edu', 
                        'socialize', 'equity', 'socio_health_pc',
                        'age_centered', 'age_centered_sq', 'class']]
        
        try:
            # 有序Logit模型（满足平行性假设）
            model_parallel = OrderedModel(y, X).fit(method='bfgs', disp=False)
            
            # 多项Logit模型（不满足平行性假设，作为对比）
            from statsmodels.discrete.discrete_model import MNLogit
            model_non_parallel = MNLogit(y, X).fit(disp=False)
            
            # 似然比检验
            ll_parallel = model_parallel.llf
            ll_non_parallel = model_non_parallel.llf
            lr_stat = -2 * (ll_parallel - ll_non_parallel)
            df = model_non_parallel.df_model - model_parallel.df_model
            p_value = chi2.sf(lr_stat, df)
            
            print(f"似然比统计量: {lr_stat:.3f}, p值: {p_value:.4f} (分组: {group_name})")
            return lr_stat, p_value
        except Exception as e:
            print(f"执行平行性假设检验失败: {str(e)}")
            return None, None
    
    def calculate_vif(self, group_data=None, group_name="全体样本"):
        """
        计算方差膨胀因子(VIF)。
        
        Args:
            group_data (pd.DataFrame, optional): 分组数据。如果未提供，则使用全体数据。
            group_name (str, optional): 分组名称。
        
        Returns:
            pd.DataFrame: VIF结果。
        """
        if group_data is None:
            group_data = self.data.copy()
        
        X = group_data[['marital', 'survey_type', 'gender', 'edu', 
                        'socialize', 'equity', 'socio_health_pc',
                        'age_centered', 'age_centered_sq', 'class']]
        
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                               for i in range(X.shape[1])]
            
            print(f"VIF计算完成。分组: {group_name}")
            return vif_data
        except Exception as e:
            print(f"计算VIF失败: {str(e)}")
            return None
    
    def train_models(self):
        """
        训练多种机器学习模型：逻辑回归、随机森林、梯度提升树和神经网络。
        
        Returns:
            dict: 模型评估结果。
        """
        # 逻辑回归
        logreg = LogisticRegression(random_state=42, class_weight='balanced')
        logreg_params = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        }
        
        # 随机森林
        rf = RandomForestClassifier(random_state=42, class_weight='balanced_subsample')
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        # 梯度提升树
        gb = GradientBoostingClassifier(random_state=42)
        gb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
        
        # 构建模型列表
        models = [
            ("Logistic Regression", logreg, logreg_params),
            ("Random Forest", rf, rf_params),
            ("Gradient Boosting", gb, gb_params)
        ]
        
        # 训练和评估模型
        for name, model, params in models:
            try:
                pipeline = ImbPipeline([
                    ('smote', SMOTE(random_state=42)),
                    ('model', model)
                ])
                
                # 网格搜索优化参数
                grid = GridSearchCV(model, params, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
                grid.fit(self.X_train_scaled, self.y_train)
                
                # 最优模型预测
                best_model = grid.best_estimator_
                y_pred_proba = best_model.predict_proba(self.X_test_scaled)[:, 1]
                auc_score = roc_auc_score(self.y_test, y_pred_proba)
                
                # 保存结果
                self.model_results[name] = {
                    "Best Params": grid.best_params_,
                    "AUC": auc_score,
                    "Classification Report": classification_report(self.y_test, best_model.predict(self.X_test_scaled)),
                    "y_pred_proba": y_pred_proba
                }
                
                print(f"Model: {name}")
                print(f"Best Parameters: {grid.best_params_}")
                print(f"AUC: {auc_score:.4f}")
                print("Classification Report:")
                print(classification_report(self.y_test, best_model.predict(self.X_test_scaled)))
                print("-" * 50)
                
            except Exception as e:
                print(f"训练模型 {name} 时出错: {str(e)}")
        
        # 神经网络
        self.train_neural_network()
        
        return self.model_results
    
    def train_neural_network(self):
        """
        训练神经网络模型。
        """
        try:
            # 构建神经网络
            def build_model():
                model = models.Sequential([
                    layers.Dense(64, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
                    layers.Dropout(0.5),
                    layers.Dense(32, activation='relu'),
                    layers.Dropout(0.5),
                    layers.Dense(1, activation='sigmoid')
                ])
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['AUC']
                )
                return model
            
            # 训练模型
            model = build_model()
            history = model.fit(
                self.X_train_resampled, self.y_train_resampled,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=0
            )
            
            # 评估模型
            y_pred_proba_nn = model.predict(self.X_test_scaled).flatten()
            auc_nn = roc_auc_score(self.y_test, y_pred_proba_nn)
            
            # 保存结果
            self.model_results["Neural Network"] = {
                "AUC": auc_nn,
                "Classification Report": classification_report(self.y_test, (y_pred_proba_nn >= 0.5).astype(int)),
                "y_pred_proba": y_pred_proba_nn
            }
            
            print(f"Neural Network AUC: {auc_nn:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, (y_pred_proba_nn >= 0.5).astype(int)))
            
        except Exception as e:
            print(f"训练神经网络时出错: {str(e)}")
    
    def run_all_analysis(self):
        """
        运行所有分析步骤。
        """
        try:
            print("开始数据加载...")
            self.load_data()
            
            print("开始数据预处理...")
            self.preprocess_data()
            self.data.to_csv("data/newdata.csv", index=False)
            
            print("开始有序逻辑回归模型拟合...")
            self.fit_ordinal_logit_model(group_name="全体样本")
            self.fit_ordinal_logit_model(group_data=self.data[self.data['group'] == '40-50后'], group_name="40-50后")
            self.fit_ordinal_logit_model(group_data=self.data[self.data['group'] == '60-70后'], group_name="60-70后")
            self.fit_ordinal_logit_model(group_data=self.data[self.data['group'] == '80-90后'], group_name="80-90后")
            
            print("开始平行性假设检验...")
            self.perform_parallelism_test(group_name="全体样本")
            self.perform_parallelism_test(group_data=self.data[self.data['group'] == '40-50后'], group_name="40-50后")
            self.perform_parallelism_test(group_data=self.data[self.data['group'] == '60-70后'], group_name="60-70后")
            self.perform_parallelism_test(group_data=self.data[self.data['group'] == '80-90后'], group_name="80-90后")
            
            print("开始计算VIF...")
            self.calculate_vif(group_name="全体样本")
            self.calculate_vif(group_data=self.data[self.data['group'] == '40-50后'], group_name="40-50后")
            self.calculate_vif(group_data=self.data[self.data['group'] == '60-70后'], group_name="60-70后")
            self.calculate_vif(group_data=self.data[self.data['group'] == '80-90后'], group_name="80-90后")
            
            print("开始训练机器学习模型...")
            self.train_models()
            
            print("分析完成。")
        except Exception as e:
            print(f"运行所有分析时出错: {str(e)}")
    
    def get_results(self):
        """
        获取所有模型的结果。
        
        Returns:
            dict: 所有模型的结果。
        """
        return self.model_results
    
    def get_classification_report(self):
        """
        获取分类报告。
        
        Returns:
            dict: 分类报告。
        """
        return self.classification_report

if __name__ == "__main__":
    # 创建HappinessStudy对象
    happiness_study = HappinessStudy(data_path="newdata.xlsx")
    
    # 运行所有分析
    happiness_study.run_all_analysis()
    
    # 获取结果
    results = happiness_study.get_results()
    
    # 打印模型结果
    print("\n模型评估结果:")
    for model_name, result in results.items():
        print(f"{model_name}: AUC = {result['AUC']:.4f}")
# 代际婚姻幸福感研究分析

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目通过统计分析与机器学习，探究不同代际（40-50后/60-70后/80-90后）的婚姻幸福感差异，包含数据处理、有序逻辑回归建模、机器学习模型评估等全流程。

## 📌 项目特点
- ✅ 模块化设：数据处理/模型训练/结果评估封装为独立类
- ✅ 异常处理：关键步骤添加全面异常捕获
- ✅ 可复现：精确依赖版本 + 详细环境配置
- ✅ 可视化：包含ROC曲线、分类报告等结果展示

## 复现步骤

```bash
# 创建虚拟环境（推荐）
conda create -n happiness python=3.10
conda activate happiness

# 进入项目目录
cd happiness-study

# 运行主程序（生成结果）
python src/main.py


import matplotlib.pyplot as plt
import numpy as np

# 设置字体
font = {'size': 12, 'family': 'Times New Roman'}
# 如果需要使用中文字体，可以取消下面一行的注释，并注释掉上面一行
# font = {'size': 12, 'family': 'SimHei'}  # 中文字体

x = np.arange(5)
y1 = np.array([0.4681, 0.4701, 0.4606, 0.4409, 0.4411])  # R@10
y2 = np.array([0.2574, 0.2621, 0.2456, 0.2375, 0.2255])  # MRR
# 设置x轴的位置偏移，以便两个折线图能够稍微分开
x_offset1 = x
x_offset2 = x + 0.1

# 将刻度方向调整向内
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# 绘画折线图，marker设置点的样式，linestyle设置线的样式，color设置颜色
R = plt.plot(x_offset1, y1, marker='o', linestyle='-', color='#f57c6e', label='R@10')
MRR = plt.plot(x_offset2, y2, marker='o', linestyle='-', color='#71b7ed', label='MRR')

# 设置坐标轴
plt.xlabel('Dropout rate of SANFM', font)
plt.ylabel('R@K / MRR', font)

# 添加子图标题（批大小）
plt.figtext(0.5, 0.04, '(e) The impact of dropout rate', ha='center', va='center', fontsize=14, fontfamily=font['family'])

# 调整刻度位置
plt.xticks(x + 0.05, [0.1, 0.2, 0.3, 0.4, 0.5])  # Embedding size，加0.05使标签居中于数据点
# 设置背景线，alpha为线宽
plt.grid(True, alpha=0.2)

# 添加图例，ncol为图例的列数（默认为1），prop设置字体，loc设置图例位置（1为右上），edgecolor设置图例边框颜色，bbox_to_anchor微调图例位置
plt.legend(loc=1, prop=font, edgecolor='black', bbox_to_anchor=(1.01, 1.01))

# 增加底部边距
plt.subplots_adjust(bottom=0.15)

# 设置y轴展示范围
plt.ylim(0, 0.6)
plt.savefig('SANFM (dropout rate).png', dpi=800) #指定分辨率保存
plt.show()
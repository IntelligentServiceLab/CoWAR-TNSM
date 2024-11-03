import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def load_data(file_path):
    data = pd.read_excel(file_path)
    return data


def process_data(data):
    related_apis = data['related_apis'].str.split('###')
    dict_apis = {}
    for apis in related_apis:
        for api in apis:
            dict_apis[api] = dict_apis.get(api, 0) + 1
    sorted_dict = dict(sorted(dict_apis.items(), key=lambda x: x[1], reverse=True))
    return sorted_dict


def calculate_percentages(sorted_dict, first_k):
    sum_all = sum(sorted_dict.values())
    sum_first_k = sum(list(sorted_dict.values())[:first_k])
    percentage = sum_first_k / sum_all * 100
    short_head_services_percentage = first_k / len(sorted_dict) * 100
    long_tail_services_percentage = (len(sorted_dict) - first_k) / len(sorted_dict) * 100
    return sum_all, sum_first_k, percentage, short_head_services_percentage, long_tail_services_percentage


def plot_long_tail_distribution(sorted_dict, first_k):
    plt.figure(figsize=(12, 6))
    x = range(1, len(sorted_dict) + 1)
    y = sorted(sorted_dict.values(), reverse=True)
    # 使用样条插值函数进行光滑处理
    spl = make_interp_spline(x, y, k=5)
    x_smooth = np.linspace(min(x), max(x), 1000)
    y_smooth = spl(x_smooth)
    plt.plot(x_smooth, y_smooth, linestyle='-', color='#c00000')
    # 填充前first_k个区域为黄色，后面区域为绿色，颜色融合自然
    plt.fill_between(x[:first_k], y[:first_k], color='#4c4cad', label='short-head services (%.1f%%)' % short_head_services_percentage)
    plt.fill_between(x[first_k:], y[first_k:], color='#4c924c', label='long-tail services (%.1f%%)' % long_tail_services_percentage)
    # 在颜色分界区添加垂直虚线
    plt.axvline(x=first_k, color='gray', linestyle='--')
    plt.text(first_k/2, 72, 'short-head services', fontsize=13, color='black', ha='center')
    # 添加左侧双箭头和文本
    plt.annotate('', xy=(0, 70), xytext=(first_k, 70), arrowprops=dict(arrowstyle="<|-|>", color='gray', linestyle='dashed'), fontsize=13, color='black', ha='center')
    plt.text(((900 / 2)+first_k/2), 72, 'long-tail services', fontsize=13, color='black', ha='center')
    plt.annotate('', xy=(first_k, 70), xytext=(900, 70), arrowprops=dict(arrowstyle="<|-|>", color='gray', linestyle='dashed'), fontsize=13, color='black', ha='center')
    plt.xlabel('Service index', fontsize=15)
    plt.ylabel('Number of invocations', fontsize=15)
    plt.xlim(left=0, right=900)
    plt.ylim(bottom=0, top=100)
    plt.legend(fontsize=15)
    # 隐藏坐标值
    plt.xticks([])
    plt.yticks([])
    plt.savefig('long_tail_picture.png', dpi=800)  # 指定分辨率保存
    plt.show()


if __name__ == "__main__":
    file_path = "../data/mashups.xlsx"
    first_k = 250
    data = load_data(file_path)
    sorted_dict = process_data(data)
    sum_all, sum_first_k, percentage, short_head_services_percentage, long_tail_services_percentage = calculate_percentages(sorted_dict, first_k)
    print(sum_all)
    print(sum_first_k)
    print("前%d个api占有总体的调用量的百分比为：%.1f%%" % (first_k, percentage))
    plot_long_tail_distribution(sorted_dict, first_k)



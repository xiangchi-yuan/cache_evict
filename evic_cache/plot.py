import numpy as np
import matplotlib.pyplot as plt

# # 加载.npy文件
# # data = np.load('llama3-8b.npy')  # 替换 'your_file.npy' 为你的文件名
# # data = np.load('llama3-8b-lacache.npy')
# data = np.load('llama3-8b-streamingllm.npy')
#
# # 假设第一行是x坐标，第二行是y坐标
# x = data[0, :]
# y = data[1, :]
#
# # 绘制图形
# plt.figure(figsize=(10, 5))  # 可以调整图形大小
# plt.plot(x, y, linestyle='-')  # Plot line only, no markers
# plt.title('Llama3-8b-StreamingLLM.')
# # plt.title('Llama3-8b')
#
# plt.xlabel('Input Length')
# plt.ylabel('Log PPL')
# plt.grid(True)  # 显示网格
# plt.show()
#
# plt.savefig('misc/plot-streaming.png')  # 保存图形为PNG文件


data1 = np.load('llama3-8b-streamingllm(1).npy')
data2 = np.load('llama3-8b-lacache(1).npy')
data3 = np.load('llama3-8b(1).npy')

# 假设第一行是x坐标，第二行是y坐标
x1 = data1[0, :]
y1 = data1[1, :]

x2 = data2[0, :]
y2 = data2[1, :]

x3 = data3[0, :]
y3 = data3[1, :]
# 绘制散点图
plt.figure(figsize=(10, 5))  # 设置图形大小
plt.plot(x1, y1, color='red', linestyle='-', label='StreamingLLM')  # 第一个数组点用红色表示
plt.plot(x2, y2, color='blue', linestyle='-', label='LaCache')  # 第二个数组点用蓝色表示
plt.plot(x3, y3, color='orange', linestyle='-', label='Vanilla')  # 第二个数组点用蓝色表示


# 添加图例、标题和坐标轴标签
plt.title('LaCache, StreamingLLM and Vanilla Performance on Llama3-8B')
plt.xlabel('Input Length')
plt.ylabel('Log PPL')
plt.legend()  # 显示图例

# 显示图形
plt.show()
# 如果需要，保存图形
plt.savefig('misc/scatter.png', format='png', dpi=300)

# data1 = np.load('llama3-8b-streamingllm.npy')
# data2 = np.load('llama3-8b-lacache.npy')
#
# # 假设第一行是x坐标，第二行是y坐标
# x1 = data1[0, :]
# y1 = data1[1, :]
#
# x2 = data2[0, :]
# y2 = data2[1, :]
# y3 = y1 -y2
#
#
# # 计算数组中元素大于0的数量
# count_positive = np.sum(y3 > 0)
#
# # 计算数组中元素大于0的比例
# proportion_positive = count_positive / len(y3)
#
# print("元素大于0的比例是：", proportion_positive)
#
# sum_positive = np.sum(y3[y3 > 0])
#
# # 计算数组中小于0的元素的总和
# sum_negative = np.sum(y3[y3 < 0])
#
# print("大于0的元素的总和是：", sum_positive)
# print("小于0的元素的总和是：", sum_negative)
#
# # 绘制散点图
# plt.figure(figsize=(10, 5))  # 设置图形大小
# # plt.scatter(x1, y1, color='red', label='StreamingLLM')  # 第一个数组点用红色表示
# # plt.scatter(x2, y2, color='blue', label='LaCache')  # 第二个数组点用蓝色表示
# plt.scatter(x1, y3, color='blue', label='Diff. ')  # 第二个数组点用蓝色表示
#
# # 添加图例、标题和坐标轴标签
# plt.title('The PPL diff. between StreamingLLM and LaCache')
# plt.xlabel('Input Length')
# plt.ylabel('Log PPL')
# plt.legend()  # 显示图例
#
# # 显示图形
# plt.show()
#
# # 如果需要，保存图形
# plt.savefig('misc/diff.png', format='png', dpi=300)
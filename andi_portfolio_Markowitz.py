#有需要改正的地方可邮andywong1012@outlook.com告知~！
# If you think there is a place need to correct, you could email andywong1012@outlook.com to contact me ~!

#首先用国泰安数据库下载了日收盘价2019-06-01至2021-06-01的
# First of all, I downloaded the daily closing price of 2019-06-01 to 2021-06-01 by using the database of Guotai 'an
import pandas as pd

data = pd.read_excel('stock_data.xlsx')
data.head()

# 将每个股票价格与最初始的价格作比较，并据此得到之后的股价走势图
# Compare each stock price with the original price to get a chart of the price since then

# 将date列从newdata中踢出
# kick the date column out of newData
date = data.pop('date')
# data.iloc[0, :] 选取第一行的数据
# data.iloc[0, :] select the first row of data
newdata = (data / data.iloc[0, :]) * 100

#使用plotly动态可视化库
# Use the Plotly Dynamic Visualization library
from plotly.offline import  plot
import plotly.graph_objs as go

#init_notebook_mode()

stocks = ['美的集团','格力电器', '五粮液','上汽集团','贵州茅台', '伊利股份', '兴业银行', '中国平安','交通银行']


def trace(df, date, stock):
    return go.Scatter(x=date,  # 横坐标日期
                      y=df[stock],
                      name=stock)  # 纵坐标为股价与（2016年10月24日）的比值


data = [trace(newdata, date, stock) for stock in stocks]
plot(data)

# ##1.2计算不同股票的均值、协方差
# 每年有252个交易日，用每日收益率乘以252得到年华收益率。现在需要计算每只股票的收益率，在金融领域中我们一般使用对数收益率。这里体现了pandas的强大，df.pct_change()直接就能得到股票收益率
# ##Calculate the mean and covariance of different stocks
# There are 252 trading days in a year. Multiply the daily rate of return by 252 to get the chronological rate of return.Now we need to calculate the return on each stock, and in finance we usually use logarithmic returns. Pandas is so powerful that 'df.pct_change()' can get the stock return directly
import numpy as np

log_returns = np.log(newdata.pct_change() + 1)
log_returns = log_returns.dropna()
log_returns.mean() * 252

# ##1.3进行正态检验
# 马科维茨的投资组合理论需要满足收益率符合正态分布，scipy.stats库为我们提供了正态性测试函数
# - scipy.stats.normaltest 测试样本是否与正态分布不同，返回p值。

## #1.3 Perform normal tests
# Markowitz's portfolio theory needs to satisfy that the return rate conforms to normal distribution, tests whether the sample differs from the normal distribution and returns a p value.

import scipy.stats as scs

def normality_test(array):
    print('Norm test p-value %14.3f' % scs.normaltest(array)[1])


for stock in stocks:
    print('\nResults for {}'.format(stock))
    print('-' * 32)
    log_data = np.array(log_returns[stock])
    normality_test(log_data)

# 从上面的检验中，这九支股票都符合正态分布。
# From the above test, all nine stocks conform to normal distribution.

# ## 1.4 投资组合预期收益率、波动率
# 随机生成一维投资组合权重向量（长度为9，与股票数量相等），因为中国股市的不允许卖空，所以投资组合权重向量中的数值必须在0到1之间。
# ## 1.4 Expected return rate and volatility of portfolio
# Random generation of one-dimensional portfolio weight vector (length 9, equal to the number of stocks). Since short-selling is not allowed in China's stock market, the value in the portfolio weight vector must be between 0 and 1.

weights = np.random.random(9)
weights /= np.sum(weights)
weights

# 投资组合预期收益率等于每只股票的权重与其对应股票的年化收益率的乘积。
# The expected return of a portfolio is equal to the product of the weight of each stock and the annualized return of the corresponding stock.
np.dot(weights, log_returns.mean()) * 252

# 投资组合波动率（方差）
# Portfolio volatility (variance)
np.dot(weights, np.dot(log_returns.cov() * 252, weights))

# 投资组合收益的年化风险（标准差）
# Annualized risk of portfolio Return (standard deviation)
np.sqrt(np.dot(weights, np.dot(log_returns.cov() * 252, weights)))

# ## 1.5 随机生成大量的投资组合权重
# 生成1000种随机的投资组合，即权重weights的尺寸为（1000\*9）。
## # 1.5 Randomly generates a large number of portfolio weights
# Generate 1000 random portfolio weights of size (1000\*9)
import matplotlib.pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')  Comment out, refer to the article：https://blog.csdn.net/sinat_28442665/article/details/87165499

port_returns = []
port_variance = []
for p in range(1000):
    weights = np.random.random(9)
    weights /= np.sum(weights)
    port_returns.append(np.sum(log_returns.mean() * 252 * weights))
    port_variance.append(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights))))

port_returns = np.array(port_returns)
port_variance = np.array(port_variance)

# 无风险利率设定为2.3735%，以http://www.chinamoney.com.cn/chinese/sddsintigy/中21年6月1日的1年期国债收益率为无风险收益率
# set the risk-free interest rate as 2.3735%, with http://www.chinamoney.com.cn/chinese/sddsintigy/ on June 1, 21 years of a 10-year Treasury yield is the risk-free rate
risk_free = 0.023735
plt.figure(figsize=(8, 6))
plt.scatter(port_variance, port_returns, c=(port_returns - risk_free) / port_variance, marker='o')
plt.grid(True)
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')

# ### 1.5.1 投资组合优化1—夏普率最大
# 建立stats函数来记录重要的投资组合统计数据（收益，方差和夏普比）。scipy.optimize可以提供给我们最小优化算法，而最大化夏普率可以转化为最小化负的夏普率。
# ### 1.5.1 Portfolio optimization 1 -- sharpe maximum
# Build a STATS function to record important portfolio statistics (return, variance and Sharpe ratio). scipy.optimize gives us minimal optimization algorithms, and maximizing sharpe can be converted to minimizing negative sharpe.

import scipy.optimize as sco

def stats(weights):
    weights = np.array(weights)
    port_returns = np.sum(log_returns.mean() * weights) * 252
    port_variance = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
    return np.array([port_returns, port_variance, port_returns / port_variance])


# 最小化夏普指数的负值
# Minimize the negative value of the Sharpe index
def min_sharpe(weights):
    return -stats(weights)[1]

# 给定初始权重
# Given the initial weight
x0 = 9 * [1. / 9]

# 权重（某股票持仓比例）限制在0和1之间。
# Weight (the proportion of a stock held) is limited to 0 and 1.
bnds = tuple((0, 1) for x in range(9))

# 权重（股票持仓比例）的总和为1。
# The total weight (proportion of shares held) is 1.
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# 优化函数调用中忽略的唯一输入是起始参数列表(对权重的初始猜测)。我们简单的使用平均分布。
#The only input ignored in the optimizer function call is the initial argument list (the initial guess about the weight).We simply use the average distribution.
opts = sco.minimize(min_sharpe,
                    x0,
                    method='SLSQP',
                    bounds=bnds,
                    constraints=cons)
opts

# 最优投资组合权重向量，小数点保留3位
# Optimal portfolio weight vector with 3 decimal places reserved
opts['x'].round(3)

# sharpe最大的组合3个统计数据分别为：
# The three statistics of sharpe's largest combination are as follows:
stats(opts['x']).round(3)

# ### 1.5.2 投资组合优化2——方差最小
# 通过方差最小来选出最优投资组合。
# ### 1.5.2 Portfolio Optimization 2 -- Minimum variance
# Select the optimal portfolio by minimizing variance.

# 定义一个函数对方差进行最小化
# Define a function to minimize the difference
def min_variance(weights):
    return stats(weights)[1]


optv = sco.minimize(min_variance,
                    x0,
                    method='SLSQP',
                    bounds=bnds,
                    constraints=cons)
optv

# 方差最小的最优投资组合权重向量
# Optimal portfolio weight vector with minimum variance
optv['x'].round(3)

#得到的投资组合预期收益率、波动率和夏普指数
# Obtained portfolio expected return, volatility and Sharpe index
stats(optv['x']).round(3)

# ### 2.5.3 组合的有效边界
# 有效边界是由一系列既定的目标收益率下方差最小的投资组合点组成的。在最优化时采用两个约束，1.给定目标收益率，2.投资组合权重和为1。
# ### 2.5.3 Effective bounds for combinations
# The efficient boundary is composed of a series of portfolio points with the smallest variance at a given target rate of return.Two constraints are used in optimization: 1. Given target rate of return; 2. Sum of portfolio weight is 1.
def min_variance(weights):
    return stats(weights)[1]


# 在不同目标收益率水平（target_returns）循环时，最小化的一个约束条件会变化。
# A constraint on minimization changes at different target_returns levels.
target_returns = np.linspace(0.0, 0.5, 50)
target_variance = []
for tar in target_returns:
    # 给定限制条件：给定收益率、投资组合权重之和为1
    cons = ({'type': 'eq', 'fun': lambda x: stats(x)[0] - tar}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    res = sco.minimize(min_variance, x0, method='SLSQP', bounds=bnds, constraints=cons)
    target_variance.append(res['fun'])

target_variance = np.array(target_variance)

# 最优化结果的展示。
# Display of optimization results.
# 叉号：构成的曲线是有效前沿（目标收益率下最优的投资组合）
# cross: The curve formed is the efficient frontier (the optimal portfolio at the target rate of return)
# 红星：sharpe最大的投资组合
# Red Star: Sharpe's largest portfolio
# 黄星：方差最小的投资组合
# Yellow Star: the portfolio with the least variance
plt.figure(figsize=(8, 4))
# 圆点：随机生成的投资组合散布的点
# dot: Randomly generated portfolio scattered points
plt.scatter(port_variance, port_returns, c=port_returns / port_variance, marker='o')
# 叉号：投资组合有效边界
# cross: portfolio efficient boundary
plt.scatter(target_variance, target_returns, c=target_returns / target_variance, marker='x')
# 红星：标记夏普率最大的组合点
# Red Star: Marks the combination point with the highest sharpe ratio
plt.plot(stats(opts['x'])[1], stats(opts['x'])[0], 'r*', markersize=15.0)
# 黄星：标记方差最小投资组合点
# Yellow Star: Mark portfolio points with minimum variance
plt.plot(stats(optv['x'])[1], stats(optv['x'])[0], 'y*', markersize=15.0)
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()

# 从黄色五角星到红色五角星是投资最有效的组合，这一系列的点所组成的边界就叫做投资有效边界。这条边界的特点是同样的风险的情况下获得的收益最大，同样的收益水平风险是最小的。从这条边界也印证了风险与收益成正比，要想更高的收益率就请承担更大的风险，但如果落在投资有效边界上，性价比最高。
# From yellow pentacle to red pentacle is the most efficient combination of investment. The boundary formed by this series of points is called the efficient investment boundary.This boundary is characterized by the maximum return for the same risk and the minimum risk for the same level of return.This boundary also proves that risk is proportional to income. If you want a higher rate of return, you need to take more risks. However, if you fall on the effective boundary of investment, the cost performance is the highest.




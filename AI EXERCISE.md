# 關於周蔚的 Scikit-Learn 的實作記錄 

找了很多的sklearn的實作，但是
[始終沒找到資料集 >_<](https://hackmd.io/7DdCE5JCQAOhiblnPWx4UQ)

所以，我決定找一個小的地方著手，找到一份很陽春清楚的算法地圖，如下：

![](https://i.imgur.com/OYDQhUM.png)（資料來源：SIGAI微信公眾號）


想要更加了解有監督學習中的線形模型中的分類方法-->也就是SVM，因為我們之前有學長就是做SVM，之前花過一段時間學習，但是一直卡卡，想這次將成功路上的絆腳石清理一下嗯~

實作中重點在探索如何使用sklearn中的SVM，沒有導入真實的數據集，打算比較熟悉后再導入數據集。


SVM（支持向量機）是為了解決分類的問題，
SVM通過將向量映射到更高維空間，找到一個最優的間隔超平面（離兩類樣本最遠的線）

## 關於線性的SVM
```
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.linear_model import LinearRegression
from scipy import stats
import pylab as pl
seaborn.set()

from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.60)

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')

# 其实随意给定3组参数，就可以画出3条不同的直线，但它们都可以把图上的2类样本点分隔开
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)

plt.xlim(-1, 3.5)
plt.show()

```
![](https://i.imgur.com/7kqR7qx.png)

這三條直線都可以將兩類樣本分開，我們需要找出它們中離樣本最遠的線。以上都是懂得，但是跟著往下走的時候，程式碼有點繞不過去。下面的內容就圖的部分介紹svm如何進行。

![](https://i.imgur.com/NnQDePb.png)

將分類邊界和支持向量和樣本點繪製。支持向量指的是被圈出來的點是離平面最近的點。

![](https://i.imgur.com/hdr7a9G.png)

不同的超平面的支持向量不一樣。SVM就是找出離樣本最遠的 超平面。

大家有興趣可以看下我的參考文檔~~:[link](https://github.com/youngxiao/SVM-demo)
謝謝大家~

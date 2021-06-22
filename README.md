
>### <font color=red>**一、高斯过程**</font>
<font size=4>$SE协方差函数=exp(\frac{-(x_1-x_2)^2}{2l^2})$</font>\
![avatar](https://github.com/ShuoLiu-Max/Bayesian-optimization/blob/main/images/gp_sample.png)
****
>### **由观测数据集,得到新样本的均值和方差**
![avatar](https://github.com/ShuoLiu-Max/Bayesian-optimization/blob/main/images/sampled.png)

**协方差函数：**
<font size=4>$$k(r\_2,l)=exp(\frac{-r\_2}{2l^2})$$</font>

$$cov(y_i,y_i)=\begin{cases} \sigma^2_fk(x_i,x_i)+\sigma^2_{noise},&if  i=j \\\ \sigma^2_fk(x_i,x_j),&otherwise\end{cases}$$

已知：
$$\begin{bmatrix}X\\\Y\end{bmatrix}\sim N(\begin{bmatrix}\mu_x \\\ \mu_y\end{bmatrix},\begin{bmatrix}A & C\\\C^T & B\end{bmatrix})$$
则有：
$$X\sim N(\mu_x,A)$$
>$$Y|X\sim N(\mu_x+C^TA^{-1}(X-\mu_y)(X-\mu_x), B-C^TA^{-1}C)$$

则对于已知观测数据$D=\{(X_i,F_i)|i=1,2,...,n\}$,
其中$F_i=f(X_i)$,对于新采样的数据$x_*$,则
$$f_\*=f(x_\*)$$

$$\begin{bmatrix}F\\f_*\end{bmatrix}\sim N(m_0I,\begin{bmatrix}K & k_*\\k^T_* & k(x_*,x_*)\end{bmatrix})$$
那么：
$$f_*|x_*,D\sim N(\mu(x_*),\sigma^2(x_*))$$
其中，
>$$\mu(x_*)=m_0+k^T_*K^{-1}(F-m_0I)$$
>$$\sigma^2(x_*)=k(x_*,x_*)-k^T_*K^{-1}k_*$$
****
>### <font color=red>**二、贝叶斯优化</font>**
><font color=yellow size=2>***1、选取采样函数***</font>\
**GP-UCB**\
$$GP-UCB(x_*)=\mu(x_*)+(1 \times 2\times(\frac{ log(|D|\times t^2 \times \pi^2)}{6\times \delta}))^{0.5}\times \sigma(x_*),\space\space\space\space\space\space\delta\in(0,1)$$
其中，$|D|$表示对$x$的定义域$D$进行离散化取值得到的点的数量。比如对于1维的情况， $D=[0,1]\subset R$，每隔 0.01 取一个$x$值，则$|D|=100$。$t$为迭代次数，即采样进度。

><font color=yellow size=2>***2、根据高斯过程，求每个点的对应的均值和方差***</font>\
在采样的取值范围空间进行等间距取点（取点个数自定义），并根据高斯过程分别计算出所取点处的均值$\mu$和方差$\sigma$。

><font color=yellow>***3、根据采样函数获得采样点***</font>\
带入到采样函数中，获得一个采样点。将该采样点加入到观测到的样本中，继续下一次采样。

所得结果如下图所示：\
![avatar](https://github.com/ShuoLiu-Max/Bayesian-optimization/blob/main/images/byes.png)
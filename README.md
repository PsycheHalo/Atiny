# Atiny Optimizer
Atiny is a gradient based PyTorch optimizer. Atiny uses half of Adam's memory to achieve results that are not inferior to Adam.

![image](./rendering.png)
![image](./renderingZoom.png)

## 思路说明
不使用二阶动量,而是使用一阶动量与当前梯度计算每个参数的更新补偿.
内置了一个学习率衰减器,自动为每组参数个性化的学习率衰减.(需要参数:ldr)
使用ArcSinh函数影响向量模长实现权重衰减

## Demo程序
demo程序在test.py文件中,直接运行即可得到此页面中Atiny与Adam的对比图.(依赖visdom)
demo中构建了一个神经网络用简陋的方式对一个包含动态随机参数的公式生成的周期性曲线进行预测.

## Install
```bash
pip install Atiny
```

## Use
```python
from Atiny import Atiny
...
optimizer=Atiny(moduel.parameters(),lr=lr,ldr=ldr,weight_decay=weight_decay)
```

## HomePage
<https://github.com/PsycheHalo/Atiny/>

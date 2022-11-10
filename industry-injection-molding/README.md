# 第四届工业大数据创新竞赛-注塑成型

The data can be downloaded from [BaiduYun](https://github.com/LongxingTan/Data-competitions/issues/3)

<!-- ![rank](./rank.png) -->

## Key of this competition:
- Solid understanding of data, feature and models are necessary
- Not a time series problem, but still need to consider the extrapolation problem for unknown scenario
- How to treat the uncertainty between local validation and LB, and how to choose one result as PB result

## Key of my solution:
- Create a custom feature from domain knowledge to evaluate the distribution difference between training and testing dataset
- Use linear regression model with above feature to modify the last novel batch's prediction result to align the test and train data
- Most important is I am lucky enough

## Other top solutions
- https://github.com/Xinchengzelin/InjectionMoulding
- https://github.com/chuangwang1991/VirtualMeasurement_molding

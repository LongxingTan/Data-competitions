# 2021阿里云供应链大赛—需求预测及单级库存优化

## How to use it
1. Download the data from [Tianchi](https://tianchi.aliyun.com/competition/entrance/531934/information) or [BaiduYun](https://github.com/LongxingTan/Data-competitions/issues/3) , put it in tcdata folder
2. Train and inference

```
sh run.sh
```

## Roadmap
1. This is a complex but also simple competition. The task is complex, but my solution to get the 2nd place is simple
2. The key is that I use a faultless ensemble method to set the safety stock, which considered weekly and daily fluctuation, you can see the details [here](https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-supply-chain/code/predict.py)
3. The demand prediction is a single LGB model, which shows obvious progress compared to rule method

## Brainstorming
- An end to end NN solution is always attractive for this task


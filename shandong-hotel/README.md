# 山东省第三届数据应用创新创业大赛-后疫情时代的民宿空置房预测

## How to use it
Download the data first, prepare the data and then run rule model.
```
cd dataset
python prepare_data.py
cd ../rule
python rule_v1.py
```

## Roadmap
It's a typical data mining task, my LGB model is not as good as my rule actually. My main steps are as below to adjust the post-process rule to predict the probability.
- use the history order information during the target date
- consider the festival pattern based on last festival
- consider the difference between Saturday and Sunday
- consider which date to calculate the base order ratio



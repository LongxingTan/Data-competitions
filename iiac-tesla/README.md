# 第二届工业互联网算法大赛-汽车赛道

The task is to predict the static torque from the dynamic sequence torque of Tesla. <br>
The code is not allowed to open source according to the rules, so I will briefly introduce my solution.

## Key of this competition
- Explore tha data, find the pattern manually, transfer the pattern into feature, and train a model
- Segment a sequence data into production procedure, and capture feature by each

## Key of my solution
- Handle a special case by a special feature, when the static torque is higher than dynamic torque.
- All prediction result add a constant factor to reduce the bias according to the local validation. (what? you want to hear the truth? the truth is, of course the LB is higher after adding a constant number)

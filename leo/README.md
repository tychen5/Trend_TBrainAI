# Leo #
台大資管所 ANTS lab (Prof. Yeali S. Sun)
r06725035@g.ntu.edu.tw

# Work Allocation & Approach

* csv/duration_discreteRate.csv => 已經完成各fid的最初發現時間與最終發現時間的差計算，也完成離散率的平均計算。

* csv/fid_avg_infectedRate.csv => 各fid所對應到的所有pid加總平均。

* csv/cid_avg_infectedRate.csv => 各fid所對應到的所有cid加總平均，若沒有出現在testing的則濾除不計入加總平均，若最終的fid仍是NaN則進行所有fid的average padding。

* csv/cid_padCid_padFid.csv => 先計算全部cid的average，給予僅出現再testing的cid賦值padding。對fid僅考慮有出現在testing的計算cid的加權平均，對最終仍為NaN的賦予fid avg padding值。

* csv/duration_statistic1.csv => 第一種time統計算法:依據百分位25,50,75減掉最小值來做為q1,q2,q3,以及timestep的min,max,std,mean。

* csv/duration_statistic2.csv => 第二種time duration統計算法:依據step來計算同fid中前後time的差，直接統計其q1,q2,3,mean,std,min,max。

* csv/MF_cid30.csv => 存放frank的mf_fid_cid_basis.pkl對照回fid的csv格式檔案

* csv/MF_cid15.csv => 上述15維度的fid對照cid之MF的csv格式

* csv/MF_pid5.csv => mf_fid_pid_basis.pkl轉換成csv可train格式

* csv/MF_pid10.csv => mf_fid_pid_basis.pkl.10

* csv/MF_cid50.csv => mf_fid_cid_basis.pkl.50

* csv/MF_pid_ratio_10.csv => pid_10 ratio

* csv/MF_cid_ratio_30.csv => cid 30 ratio

## Intro of Me ##
- skilled in malware analysis, deep learning / machine learning

## Background of Me ##
- 網路資安lab: malware analysis、5G IoT technology
- malware behavior analysis


## Ongoing ##
- DNN + softmax -> binary classification / regression

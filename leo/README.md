# Leo #
hello there~

* csv/duration_discreteRate.csv => 已經完成各fid的最初發現時間與最終發現時間的差計算，也完成離散率的平均計算。

* csv/fid_avg_infectedRate.csv => 各fid所對應到的所有pid加總平均。

* csv/cid_avg_infectedRate.csv => 各fid所對應到的所有cid加總平均，若沒有出現在testing的則濾除不計入加總平均，若最終的fid仍是NaN則進行所有fid的average padding。

* csv/cid_padCid_padFid.csv => 先計算全部cid的average，給予僅出現再testing的cid賦值padding。對fid僅考慮有出現在testing的計算cid的加權平均，對最終仍為NaN的賦予fid avg padding值。

## Intro of Me ##
- 孫媽lab

## Background of Me ##
- 網路資安lab: malware analysis、5G IoT technology
- know a little of DL
- new to ML

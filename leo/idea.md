# IDEA #
## Feature ##
### timestamp ###
1. 計算fid第一次被發現到最後發現該fid的時間=>

### product ###
1. 計算各pid的偵測率=>意即:統計該pid所有fid部分 出現1的次數/1+0的次數

### cutomer ###
1. 各個cutomer確診為中毒的機率


## Method ##
### DL ###
- 對cid、pid進行encoding(不知道是要編碼就好還是要使用one-hot)
- 以fid作為基準，將同樣fid的cid

### DL-Regression ###
- 變成0跟1的scalar作為label
- 缺點: training時沒有中間值，可能loss會大或是答案太sharp而train不起來

### DL-binary classification ###
- 0為false，1為true
- 缺點: output不是scalar，估計不太可行



## 其他補充說明 ##
- 因為沒有什麼DL以外的ML domain knowhow，所以其實不太知道要怎樣才叫做feature，就是怎樣把feature弄到ML當中
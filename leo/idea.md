# IDEA #
## Feature ##
fid搭配其他attriubute
### time-stamp ###
1. 計算fid第一次被發現到最後發現該fid的時間=>分析大約是在多少時間內的(或超過多少時間的)可能為中毒的機率有多高、尖峰維持多久?多密集?有多少?
	- sparse rate: 出現總次數/DURATION
2. 某一段時間內的該fid出現次數
3. 平均隔多久出現下一個

### product ###
1. 計算各pid的偵測率=>意即:統計該pid所有fid部分 出現1的次數/1+0的次數
2. 有幾種產品偵測該fid是可疑的

### customer ###
1. 各個cutomer確診為中毒的機率 => 該fid的平均cid中毒率多少


## Method ##
### DL ###
- 對cid、pid進行encoding(不知道是要編碼就好還是要使用one-hot)
- 將各feature合成np array過fully-connected

### DL-CNN ###
- 因為每筆fid所具有的比數不同，所以可能需要padding，有的差距甚大感覺不太容易
- X

### DL-RNN ###
- 同樣fid的(cid+pid)n個作為一筆訓練資料，所以對應的label也就只有一個0或1
- O

### DL-Regression ###
- 變成0跟1的scalar作為label
- 缺點: training時沒有中間值，可能loss會大或是答案太sharp而train不起來
- O

### DL-binary classification ###
- 0為false，1為true
- 缺點: output不是scalar，估計不太可行
- X



## 其他補充說明 ##
- 因為沒有什麼DL以外的ML domain knowhow，所以其實不太知道要怎樣才叫做feature，就是怎樣把feature弄到ML當中
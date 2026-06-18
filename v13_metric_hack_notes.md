# v13 Metric Hack Notes

璁板綍鏃堕棿锛?026-06-18

## 鐗堟湰瀹氫綅

v13 涓嶆槸鏂拌缁冩ā鍨嬬増鏈紝鑰屾槸鍦ㄥ綋鍓嶇嚎涓婃渶濂芥€濊矾涓婂姞涓€涓畼鏂瑰厑璁哥殑 metric-hack 鍚庡鐞嗐€?
鍩虹妯″瀷锛?
```text
LightGBM + XGBoost + CatBoost blend
weights = 0.45 / 0.45 / 0.10
```

杩欑粍鏉冮噸瀵瑰簲 public 琛ㄧ幇鏇村ソ鐨?v11 鎬濊矾銆倂12 鐨?OOF 鏉冮噸涓?`0.30 / 0.60 / 0.10`锛宲ublic 涓?`0.55885`锛岀暐浣庝簬 v11 鐨?`0.56081`锛屾墍浠?v13 鍥炲埌 v11 鏉冮噸銆?
## Metric Hack 閫昏緫

鍙傝€冪涓€鍚嶆柟妗堣鏄庯細

```python
DIVIDE = 1 / 2
REDUCE = 0.03
condition = WEEK_NUM < (max(WEEK_NUM) - min(WEEK_NUM)) * DIVIDE + min(WEEK_NUM)
score[condition] = clip(score[condition] - REDUCE, 0, 1)
```

鏈」鐩?v13 浣跨敤锛?
```text
divide = 0.5
reduce = 0.03
```

瀹炵幇缁嗚妭锛?
- 浼樺厛鐩存帴浣跨敤 `test_base.WEEK_NUM` 浣滀负鏃堕棿淇″彿銆?- 濡傛灉 `WEEK_NUM` 涓嶅彲鐢紝鍒?fallback 鍒?`credit_bureau_a_1.refreshdate_3813885D` 鎭㈠ week銆?- 鍙湪棰勬祴瀹屾垚鍚庝慨鏀?`submission.score`锛屼笉褰卞搷鐗瑰緛宸ョ▼鍜屾ā鍨嬫帹鐞嗐€?
## 鏂囦欢

```text
notebook:
submission/v13_inference_metric_hack.ipynb

artifact:
submission/artifact_v13/

local smoke output:
outputs/v13_metric_hack_smoke_submission.csv
```

## 鏈湴 Smoke Test

鏈湴 sample test 鍙湁 10 琛岋紝骞朵笖 `WEEK_NUM` 鍏ㄩ儴涓?100锛屾墍浠ヤ笉浼氳Е鍙戣皟鏁达細

```text
time_signal_non_null 10
time_signal_min      100
time_signal_max      100
adjusted_rows        0
```

hidden test 璺ㄥ畬鏁存椂闂存锛屾墠浼氭湁鍓嶅崐娈垫牱鏈涓嬭皟銆?
## 褰撳墠鍒ゆ柇

杩欐槸涓€涓悗澶勭悊璧屽崥椤癸紝涓嶄唬琛ㄦā鍨嬫湰韬洿寮恒€傚畼鏂圭涓€鍚嶄篃璇存槑 private 鏈€浼樺弬鏁板ぇ姒傜巼鍦ㄤ竴涓尯闂村唴锛宍divide=0.5, reduce=0.03` 鏄腑鎬ч€夋嫨銆?
濡傛灉 v13 public/private 鍙樺ソ锛屽彲浠ョ户缁洿缁?`divide` 鍜?`reduce` 鍋?v15/v16 灏忚寖鍥村弬鏁板疄楠岋紱濡傛灉鍙樺樊锛岀洿鎺ュ洖鍒?v11/v12 妯″瀷铻嶅悎涓荤嚎銆?

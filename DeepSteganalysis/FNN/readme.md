---
title: readme
date: 2017/12/1  20:31
tags: 
---

## 模型

[Deep learning is a good steganalysis tool when embedding key is reused for different images, even if there is a cover sourcemismatch](http://www.lirmm.fr/~chaumont/publications/IST_ELECTRONIC_IMAGING_Media_Watermarking_Security_Forensics_2016_PIBRE_PASQUET_IENCO_CHAUMONT_deep_steganalysis_draft.pdf)

![图片描述](http://otivusbsc.bkt.clouddn.com/99670aa9-4884-4b9b-bbe4-eace069820d9)
其中, $F^{(0)}$ 为

```python
        F0 = np.array([[-1, 2, -2, 2, -1],
                       [2, -6, 8, -6, 2],
                       [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]], dtype=np.float32)
```
## 原模型

data无处理, 无BN层


<!-- more -->

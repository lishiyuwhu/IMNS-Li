---
title: readme
date: 2017/12/28 16:34
tags: 
---

# CroppedBossBase-1.0-256x256_SUniward0.4bpp

512×512均等切割为四个256×256

使用相同的key做隐写

Q. When you prepared the database of 256x256 images from BOSSbase, did you embed them after cropping (embedding into 256x256 images) or did you embed the payload into the full size 512x512 and then cropped them into four?

A. We have embeded in the 256x256 images from the cropped BOSSBase.

Q. What was the value of the stabilizing constant that is used in S-UNIWARD to prevent dividing by zero in computing the embedding costs?

A. The sigma is equal to 1 as recommanded in the fixed version.

Q. Which version of S-UNIWARD did you use in your experiments?

A. For S-Uniward, the C++ linux version has been used. *The same key * has been used for embedding and embedding has been done with the simulator.

 - Source

 http://www.lirmm.fr/~chaumont/SteganalysisWithDeepLearning.html


#



<!-- more -->
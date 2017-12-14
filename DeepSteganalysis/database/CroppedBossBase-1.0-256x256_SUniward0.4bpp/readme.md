# The databases:

 - The cropped BOSSBase database : 
 CroppedBossBase-1.0-256x256_cover.rar.
 - The cropped BOSSBase stego images with S-UNIWARD at 0.4 bpp :  
   CroppedBossBase-1.0-256x256_stego_SUniward0.4bpp.rar.

# Q&A

Q. When you prepared the database of 256x256 images from BOSSbase, did you embed them after cropping (embedding into 256x256 images) or did you embed the payload into the full size 512x512 and then cropped them into four? 

A. We have embeded in the 256x256 images from the cropped BOSSBase. 

Q. What was the value of the stabilizing constant that is used in S-UNIWARD to prevent dividing by zero in computing the embedding costs?

A. The sigma is equal to 1 as recommanded in the fixed version. 

Q. Which version of S-UNIWARD did you use in your experiments?

A. For S-Uniward, the C++ linux version has been used. *The same key * has been used for embedding and embedding has been done with the simulator.

---

#Source

http://www.lirmm.fr/~chaumont/SteganalysisWithDeepLearning.html
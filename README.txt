DJICA (Vanilla)

Status: working

Improvements to be done:  

1_ Optimizing the messages passed between the Remote and the locals
2_ Tuning the GD algorithms. The results are not comparable with SciKit learn fastICA (centralized)
3_ Random selection of local data for local gradients has not been yet performed, thus Vanilla version. 


Note: 
At the moment the resulting Amari ISI performance metric for djICA with perfect PCA subspace matrix U, is about 0.2 more than what fastICA (centralized) yields. 

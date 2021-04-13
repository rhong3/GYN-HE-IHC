# Created by: RH
# Created on: 4/13/21

library(ggplot2)
file = read.csv("../align/align_eval_full.csv")
median(file$Align_score)

png("../align/align_eval.png", units="in", width=5, height=5, res=300)
ggplot(file, aes(x=Align_score)) +
  geom_histogram(bins=20) + geom_vline(aes(xintercept = median(file$Align_score)),col='red',size=1) + 
  theme_bw()+ xlab("Align score (median=0.916875)")
dev.off()

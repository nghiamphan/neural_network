library(scales)

data = read.table("data_banknote_authentication.txt", header = FALSE, sep = ",", dec = ".")

par(cex.axis=0.8)

boxplot(data$V1, data$V2, data$V3, data$V4, 
        names = c('variance of Wavelet\nTransformed image', 'skewness of Wavelet\nTransformed image', 'curtosis of Wavelet\nTransformed image', 'entropy of image'), 
        xlab = 'Features', 
        main='Without normalization')

data$V1 <- rescale(data$V1, to = c(-1, 1))
data$V2 <- rescale(data$V2, to = c(-1, 1))
data$V3 <- rescale(data$V3, to = c(-1, 1))
data$V4 <- rescale(data$V4, to = c(-1, 1))

boxplot(data$V1, data$V2, data$V3, data$V4, 
        names = c('variance of Wavelet\nTransformed image', 'skewness of Wavelet\nTransformed image', 'curtosis of Wavelet\nTransformed image', 'entropy of image'), 
        xlab = 'Features', 
        main='With normalization')

data = H5Fopen("aod_r.h5")
data = data$r
data = t(data$block0_values[1:14400,1:3652])
train_data = data[2657:3387, seq(1, 14400, 50)] / 10000
test_data = data[3288:3652, seq(1, 14400, 50)] / 10000
model_50 = VAR(train_data, p=2)

const = model_50$coef[1,]
weights = model_50$coef[2:dim(model_50$coef)[1],]
rmse.list = data.frame(matrix(nrow = 364 - 1, ncol = 1))
offsets = c(1,2,3,4,5,6,7,8,9,10)
mean.list = data.frame(matrix(nrow=5, ncol=1))


########### VAR Model Prediction ##########
for (off in offsets) {
    
    for (i in 1:(364 - off)) {
        pred = as.numeric(test_data[i,])
        for (k in 1:(off+1)) {
            pred = pred %*% weights 
        }
        rmse.list[i, ] = sqrt(sum((pred - test_data[i + off,])^2) / dim(model_50$coef)[2])
    }
    
    print(mean(rmse.list[,1]))
    mean.list[off,] = mean(rmse.list[,1])
    rmse.list = data.frame(matrix(nrow = 364 - off - 1, ncol = 1))
}

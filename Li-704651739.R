#Dennis Li
#704651739


data <- read.csv("http://www.stat.ucla.edu/~jsanchez/data/hwk6data.csv")
names(data)[1] <- "hs"
names(data)[2] <- "uw"
names(data)[3] <- "ur"
data <- ts(data, start = c(1959,1), frequency = 12)
data.train <- ts(data[1:704,], start = c(1959, 1), frequency = 12)
data.test <- ts(data[705:716,], start = c(2017, 9), frequency = 12)


###########################
###### Section 3 ##########
###########################

hs <- ts(data.train[,1], start = c(1959, 1), frequency = 12)
uw <- ts(data.train[,2], start = c(1959, 1), frequency = 12)
ur <- ts(data.train[,3], start = c(1959, 1), frequency = 12)

hs.decom.additive <- decompose(hs)
uw.decom.additive <- decompose(uw)
ur.decom.additive <- decompose(ur)

hsObserved <- hs.decom.additive$x
uwObserved <- uw.decom.additive$x
urObserved <- ur.decom.additive$x

hsSeasonal <- hs.decom.additive$seasonal
uwSeasonal <- uw.decom.additive$seasonal
urSeasonal <- ur.decom.additive$seasonal

hsTrend <- hs.decom.additive$trend
uwTrend <- uw.decom.additive$trend
urTrend <- ur.decom.additive$trend

hsRandom <- hs.decom.additive$random
uwRandom <- uw.decom.additive$random
urRandom <- ur.decom.additive$random

#decomposition
par(mfrow = c(3,3))
plot(hsObserved, main = "Plot for Observed hs")
plot(uwObserved, main = "Plot for Observed uw")
plot(urObserved, main = "Plot for Observed ur")
plot(hsTrend, main = "Plot for Trend hs")
plot(uwTrend, main = "Plot for Trend uw")
plot(urTrend, main = "Plot for Trend ur")
plot(hsRandom, main = "Plot for Random hs")
plot(uwRandom, main = "Plot for Random uw")
plot(urRandom, main = "Plot for Random ur")
dev.off()

#seasonal box plots
par(mfrow = c(3,1))
boxplot(hs~cycle(hs), main = "Box Plot for hs seasonal")
boxplot(uw~cycle(uw), main = "Box Plot for uw seasonal")
boxplot(ur~cycle(ur), main = "Box Plot for ur seasonal")
dev.off()

#time series plots
par(mfrow = c(3,1))
plot.ts(hs, main = "Time Plot of hs", ylab = "housing starts")
plot.ts(uw, main = "Time Plot of uw", ylab = "unemployment rate % (women)")
plot.ts(ur, main = "Time Plot of ur", ylab = "civilian unemployment rate %")
dev.off()

#acf/ccf
acf(data.train)

#unit root tests/cointegration
library(tseries)
adf.test(hs)
adf.test(uw)
adf.test(ur)
po.test(cbind(hs,uw,ur))


#volatile check
acf(hs-mean(hs), main = "ACF of hs")
acf((hs-mean(hs))^2, main = "ACF of squared mean adjusted hs")
  
acf(uw-mean(uw), main = "ACF of uw")
acf((uw-mean(uw))^2, main = "ACF of squared mean adjusted uw")

acf(ur-mean(ur), main = "ACF of ur")
acf((ur-mean(ur))^2, main = "ACF of squared mean adjusted ur")


###########################
###### Section 4 ##########
###########################

acf(hs, main = "ACF of hs")

reg.diff <- diff(hs, lag = 1, diff = 1)
par(mfrow = c(1,2))
acf(reg.diff, main = "ACF of differenced hs")
pacf(reg.diff, main = "PACF of differenced hs")
dev.off()

seas.diff <- diff(hs, lag = 12, diff = 1)
par(mfrow = c(1,2))
acf(seas.diff, main = "ACF of seasonally differenced hs")
pacf(seas.diff, main = "PACF of seasonally differenced hs")
dev.off()

seas.reg.diff <- diff(reg.diff, lag = 12, diff = 1)
par(mfrow = c(1,2))
acf(seas.reg.diff, main = "ACF of seas+reg differenced data")
pacf(seas.reg.diff, main = "PACF of seas+reg differenced data")
dev.off()

par(mfrow=c(3,1))
acf(reg.diff,lag=50, main="reg only diff")
acf(seas.diff,lag=50,main="seas only diff")
acf(seas.reg.diff,lag=50,main="reg and seas diff")
dev.off()



#SARIMA model
sarimaModel <- arima(hs, order = c(0,1,1), seas = list(order = c(1,1,1), 12))
sarimaModel
acf(sarimaModel$residuals, main = "ACF of residuals")
acf((sarimaModel$residuals)^2, main = "ACF of residuals squared", lag = 50)

garch1 <- garch(sarimaModel$residuals, order = c(1,1), trace = F)
garch1
acf(garch1$residuals, na.action = na.pass, main = "ACF of residuals w/ GARCH")
acf(garch1$residuals^2, na.action = na.pass, main = "ACF of residuals squared w/ GARCH")

#forecasting w/ SARIMA
y.pred1 <- ts(predict(sarimaModel, n.ahead = 12, se.fit = TRUE))
y.pred1
cil = ts((y.pred1$pred - 1.96 * y.pred1$se), start = c(2017,9), frequency = 12)
cil
ciu = ts((y.pred1$pred +1.96 * y.pred1$se), start = c(2017,9), frequency = 12)
ciu
ts.plot(cbind(hs, y.pred1$pred, cil, ciu), lty = c(1, 2, 3, 3),
        col=c("blue", "green", "red","red"), ylab="y_t", main = "Sarima(0,1,1,)(1,1,1)12 Forecast")


#RMSE
MSE <- sum((data.test[,1] - y.pred1$pred)^2)/12
RMSE <- sqrt(MSE)
RMSE

###########################
###### Section 5 ##########
###########################

#regression
seasonal <- factor(cycle(data.train))
times <- time(data.train)
regModel <- lm(hs ~ uw+ur + seasonal)
regModel
acf(regModel$residuals)
pacf(regModel$residuals)

#fit AR(1) to residuals
modelres1 <- arima(ts(rstudent(regModel)), order = c(1,0,0))
modelres1
acf(modelres1$residuals)

#refit with gls to account for autocorrelation
library(nlme)
glsmodel <- gls(hs~uw + ur+
                  seasonal, 
                correlation = corARMA(c(0.9189),
                                      p = 1))
glsmodel
acf(ts(residuals(glsmodel, type = "normalized")), main = "ACF of Residuals of GLS Model")

#forecast
new.dat <- data.frame(futureMonths= rep(1:12,1))
y.pred2 <- predict(glsmodel, new.dat)[1:12]
y.pred2 <- ts(y.pred2,start=c(2017,9),freq=12)
y.pred2

ts.plot(hs,y.pred2,lty=1:2, col=c("blue", "green"), main="gls Forecast",
        ylab="hs", xlab="Time")




#RMSE
MSE <- sum((data.test[,1] - y.pred2)^2)/12
RMSE <- sqrt(MSE)
RMSE


###########################
###### Section 6 ##########
###########################

library(vars)
acf(diff(data.train))




varModel = VAR(data.train, p = 3, type = "trend")
varModel
summary <- summary(varModel)
acf(resid(varModel))

which(summary[[2]]$hs$coefficients[,4] < 0.05)
which(summary[[2]]$uw$coefficients[,4] < 0.05)
which(summary[[2]]$ur$coefficients[,4] < 0.05)




#predict
VAR.pred <- predict(varModel, n.ahead=12)
VAR.pred

#extract hs
housingPred<- ts(VAR.pred$fcst$hs[,c("fcst", "lower", "upper")],
                 start = c(2017,9), frequency = 12)
housingPred
ts.plot(cbind(window(hs, start = c(1959,1)), housingPred), 
        lty = c(1,2,2,2), col=c("blue", "green", "red", "red"), main = "VAR hs Forecast w/ CI")

#RMSE
housingPred <- ts(VAR.pred$fcst$hs[,1],
                  start = c(2017,9), frequency = 12)
MSE <- sum((data.test[,1] - housingPred)^2)/12
RMSE <- sqrt(MSE)
RMSE


#impulse response analysis 
#shock hs
irf.hs = irf(varModel, impulse = "hs", response = c("hs", "uw", "ur"), boot = FALSE,n.ahead=100)
plot(irf.hs)

#shock uw
irf.uw = irf(varModel, impulse = "uw", response = c("hs", "uw", "ur"), boot = FALSE,n.ahead=100)
plot(irf.uw)

#shock ur
irf.ur = irf(varModel, impulse = "ur", response = c("hs", "uw", "ur"), boot = FALSE,n.ahead=100)
plot(irf.ur)





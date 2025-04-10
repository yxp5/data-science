# yxp5

library(readxl)
library(ggplot2)

data1 <- read_excel('Prices of gold, Data.xlsx')
df <- data.frame(year = data1[,2], price = data1[,1])
df

model <- lm(Per.Troy.ounce.in.USD ~ Year, data = df)
summary(model)

newdata <- data.frame(Year=c(2024, 2025, 2026, 2027))
predict(model, newdata)

dft <- data.frame(year = data1[42:46, 2], price = data1[42:46, 1])
dftmp <- data.frame(Year = c(2024), Per.Troy.ounce.in.USD = c(2386))
df2 <- rbind(dft, dftmp)
model2 <- lm(Per.Troy.ounce.in.USD ~ Year, data = df2)
predict(model2, newdata)
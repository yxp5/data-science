# yxp5
# Data analysis of neoadjuvant treatment effect
# on patients' white blood cell level

# 1. White Blood Cell

data1 <- read.csv('Blood Routine Counts.csv')
before1 <- data1$Before.Treatment
after1 <- data1$After.Treatment
diff1 <- after1 - before1

# Standard paired t-test

t.test(before1, after1, paired=TRUE)

# Wilcoxon signed ranked test
# To avoid extreme positive or negative value that
# may cause bias in result

wilcox.test(before1, after1, paired=TRUE)

# Both tests show no effect of variation in white
# blood cell from the treatment, but ranked-based
# method gives more confident in accepting the null

# 2. Tumor Size

data2 <- read.csv('Data for Tumor Marker Levels.csv')
before2 <- data2$Before.Treatment
after2 <- data2$After.Treatment
diff2 <- after2 - before2

t.test(before2, after2, paired=TRUE)
wilcox.test(before2, after2, paired=TRUE)

# Both tests reject the null










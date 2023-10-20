# Libraries
library(ggplot2)
library(dplyr)
library(hrbrthemes)
library(reshape2)

batch1 <- data.frame(error = c(1648.3645, 1254.5695, 1035.9487, 1266.8861,
                               1187.1301, 1236.6136, 1222.1943, 1344.2181,
                               1383.3608, 1371.6438, 1416.2802, 1344.8635,
                               1361.179, 1369.7549, 1448.9116, 1413.3579,
                               1405.9067, 1373.3335, 1368.1765, 1342.5525,
                               1327.284, 1320.4323, 1321.3368, 1351.7552,
                               1328.1216, 1323.942, 1318.3201, 1298.5973,
                               1302.4738, 1276.7152),
                     batch = rep("batch size 1", 30))

batch8 <- data.frame(error = c(1507.4119, 1109.3912, 929.14844, 1202.015,
                               1157.0432, 1246.3588, 1248.4097, 1400.6251,
                               1444.6783, 1408.2599, 1472.32, 1396.7295,
                               1410.9513, 1422.6143, 1481.1907, 1431.1797,
                               1431.1339, 1398.7969, 1394.3369, 1369.5062,
                               1345.4983, 1341.9133, 1338.6271, 1369.7375,
                               1348.4783, 1332.2739, 1329.6385, 1295.2924,
                               1309.3215, 1282.2994),
                     batch = rep("batch size 8", 30))

batch16 <- data.frame(error = c(2401.7683, 1718.8068, 1581.7368, 2095.0251,
                                2092.2974, 2258.5193, 2249.0596, 2423.8694,
                                2438.7256, 2369.9326, 2503.6567, 2399.0403,
                                2420.635, 2446.888, 2512.3164, 2420.7834,
                                2441.6914, 2408.4333, 2409.477, 2375.39,
                                2333.3164, 2328.641, 2312.4739, 2357.4644,
                                2337.2417, 2306.363, 2304.0317, 2251.069,
                                2241.2798, 2207.562),
                      batch = rep("batch size 16", 30))

batch32 <- data.frame(error = c(3123.8494, 2207.6382, 2085.3535, 2662.5784,
                                2743.212, 2968.1526, 2973.8982, 3168.5698,
                                3182.937, 3140.2222, 3299.597, 3183.2488,
                                3204.958, 3239.8752, 3277.4307, 3173.5356,
                                3211.4675, 3184.509, 3203.0254, 3166.7505,
                                3127.5488, 3110.971, 3093.592, 3142.0603,
                                3130.8894, 3106.4614, 3107.7563, 3049.855,
                                3029.566, 2988.8765),
                      batch = rep("batch size 32", 30))

batch64 <- data.frame(error = c(3557.814, 2684.4521, 2591.6895, 3198.1357,
                                3261.86, 3509.4375, 3526.9949, 3740.044,
                                3761.2412, 3678.7017, 3844.2947, 3724.7058,
                                3755.2878, 3795.685, 3826.0828, 3713.4338,
                                3753.4182, 3725.609, 3737.3232, 3702.243,
                                3652.2075, 3645.0227, 3625.6006, 3674.9229,
                                3659.0835, 3621.7173, 3622.559, 3556.7983,
                                3541.8154, 3502.7983),
                      batch = rep("batch size 64", 30))

print(mean(batch1$error))
print(mean(batch8$error))
print(mean(batch16$error))
print(mean(batch32$error))
print(mean(batch64$error))

batch_size <- rbind(batch1, batch8, batch16, batch32, batch64)

# Build the boxplot. In the 'fill' argument, give this column
s <- ggplot(batch_size, aes(x = batch, y = error, fill = batch)) +
  geom_boxplot() +
  xlab("") +
  scale_y_continuous(
                     breaks = round(seq(min(signif(batch_size$error, 1)),
                                        max(batch_size$error), by = 250), 1))

ggsave("batches.png", s)

batches <- c("1", "8", "16", "32", "64")
for (x in batches) {
  batchi <- read.csv(paste("b", x, ".txt", sep = ""), header = TRUE)
  batch <- melt(batchi, id.vars = "Epoch", variable.name = "Loss")
  s <- ggplot(batch, aes(Epoch, value)) +
    geom_line(aes(colour = Loss)) +
    ggtitle(paste("Learning curve for batch ", x, sep = "")) +
    scale_y_continuous(
                       breaks = round(seq(min(signif(batchi$Val_loss, 2)),
                                          max(batchi$Val_loss), by = 500), 1))
  ggsave(paste("batch", x, ".png", sep = ""), s)
}
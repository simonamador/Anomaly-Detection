# Libraries
library(ggplot2)
library(dplyr)
library(hrbrthemes)
library(reshape2)
library(data.table)

anomalies <- c("healthy", "vm")
views <- c("L", "A", "S")
losses <- c("L2", "SSIM", "MS_SSIM")
extras <- c("0.1_", "1.0_", "10.0_", "100.0_")

view <- "L"
extra <- "Relu_"
model <- "bVAE"
loss <- "L2"
batch <- "64"
date <- "20231106"

for (extra in extras) {

  if (view == "L") {
    l <- 110
    name <- "sagittal"
  } else if (view == "A") {
    l <- 158
    name <- "frontal"
  } else {
    l <- 126
    name <- "axial"
  }

  n_name <- paste("history_", extra, view, "_", model, "_AE_", loss,
                  "_b", batch, "_", date, sep = "")
  f_name <- paste("files/", n_name, ".txt", sep = "")

  if (file.exists(f_name)) {
    hist <- read.csv(f_name, header = TRUE)
    tc <- subset(hist, select = c(Epoch, Train_loss, Val_loss))
    metrics <- subset(hist, select = -c(Train_loss, Val_loss))

    tc <- melt(as.data.table(tc), id.vars = "Epoch",
               variable.name = "Loss")
    metrics <- melt(as.data.table(metrics), id.vars = "Epoch",
                    variable.name = "Metrics")

    tc_plot <- ggplot(tc, aes(Epoch, value)) +
      geom_line(aes(colour = Loss)) +
      ggtitle(paste("Training curve for view ", name, sep = ""))

    mt_plot <- ggplot(metrics, aes(Epoch, value)) +
      geom_line(aes(colour = Metrics)) +
      ggtitle(paste("Metrics for view ", name, sep = ""))

    ggsave(paste("graphs/tc", n_name, ".png", sep = ""), tc_plot)
    ggsave(paste("graphs/metric", n_name, ".png", sep = ""), mt_plot)
  } else {
    print("No such file.")
  }
}

for (extra in extras) {

  if (view == "L") {
    name <- "sagittal"
  } else if (view == "A") {
    name <- "frontal"
  } else {
    name <- "axial"
  }

  h_name <- paste("healthy_", extra, view, "_", model, "_AE_", loss,
                  "_b", batch, "_", date, sep = "")
  vm_name <- paste("vm_", extra, view, "_", model, "_AE_", loss,
                   "_b", batch, "_", date, sep = "")

  if (file.exists(paste("files/", h_name, ".txt", sep = ""))) {

    h_val <- read.csv(paste("files/", h_name, ".txt", sep = ""), header = TRUE)
    vm_val <- read.csv(paste("files/", vm_name, ".txt", sep = ""),
                       header = TRUE)

    h_val <- subset(h_val, select = c("mae", "mse"))
    vm_val <- subset(vm_val, select = c("mae", "mse"))

    nh <- nrow(h_val)
    ids_h <- c(1:nh)
    nvm <- nrow(vm_val)
    ids_vm <- c(1:nvm)

    h_val <- cbind(id = ids_h, h_val)
    vm_val <- cbind(id = ids_vm, vm_val)

    h_val <- melt(as.data.table(h_val), id.vars = "id",
                  variable.name = "Metric")
    h_val$dataset <- "healthy"
    vm_val <- melt(as.data.table(vm_val), id.vars = "id",
                  variable.name = "Metric")
    vm_val$dataset <- "ventriculomegaly"

    df <- rbind(h_val, vm_val)

    s <- ggplot(df, aes(x = Metric, y = value, fill = dataset)) +
      geom_boxplot() +
      ggtitle(paste("Validation metrics for ", name, " view"))
    ggsave(paste("graphs/", vm_name, ".png", sep = ""), s)
  } else {
    print("No such file.")
  }
}
# =============================================================================
# 新生儿肺炎（Pneumonia）预测模型 - 完整版（遵循参考文章思路）
# 包含：缺失值多重插补、LASSO特征选择、相关性分析、8种模型构建、
#       ROC/校准曲线/决策曲线评估、SHAP分析（最优模型）、600dpi图片保存
# =============================================================================

rm(list = ls())

# ---- 1. 加载必要的包 ----
required_packages <- c(
  "tidyverse", "tidymodels", "themis", "shapviz",
  "xgboost", "ranger", "pROC", "ggplot2", "corrplot",
  "glmnet", "kernlab", "discrim", "dcurves", "mice",
  "nnet", "keras", "tensorflow", "kknn", "rpart",
  "vip", "DALEX", "patchwork","SHAPBoost", "SHAPforxgboost"
)
installed <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!pkg %in% installed) install.packages(pkg)
  library(pkg, character.only = TRUE)
}
setwd("C:/Users/Administrator/Desktop/安岳/R/")

# ---- 2. 创建输出目录 ----
output_dir <- "NP_Model_Results_600dpi"
if (!dir.exists(output_dir)) dir.create(output_dir)

# ---- 3. 读取数据 ----

df_raw <- read.csv("GDMnew111.csv", na.strings = c("", "NA"), stringsAsFactors = FALSE)

# ---- 4. 数据清洗与变量类型定义（根据GDMnew222.csv调整）----
df_clean <- df_raw %>%
  mutate(
    Education = factor(Education, levels = c(0,1), labels = c("low","high")),
    Job = factor(Job, levels = c(0,1), labels = c("unemployed","employed")),
    DiaH = factor(DiaH, levels = c(0,1), labels = c("no","yes")),
    ThyH = factor(ThyH, levels = c(0,1), labels = c("no","yes")),
    APO  = factor(APO, levels = c(0,1), labels = c("no","yes")),
    PH   = factor(PH, levels = c(0,1), labels = c("no","yes")),
    Primipara = factor(Primipara, levels = c(0,1), labels = c("no","yes")),
    HG   = factor(HG, levels = c(0,1), labels = c("no","yes")),
    Preterm = factor(Preterm, levels = c(0,1), labels = c("no","yes")),
    Caesarean = factor(Caesarean, levels = c(0,1), labels = c("Natural","Caesarean")),
    Sex = factor(Sex, levels = c(0,1), labels = c("Female","Male")),
    Macrosomia = factor(Macrosomia, levels = c(0,1), labels = c("no","yes")),
    Pneumonia = factor(Pneumonia, levels = c(0,1), labels = c("no","yes")),   # 结局变量
    Age = as.numeric(Age),
    BMI = as.numeric(BMI)
  )

# ---- 5. 缺失值处理：删除缺失率>20%的变量，其余多重插补 ----
# 计算变量缺失率
missing_pct <- sapply(df_clean, function(x) mean(is.na(x)) * 100)
vars_to_keep <- names(missing_pct[missing_pct <= 20])
df_filtered <- df_clean %>% select(all_of(vars_to_keep))


df_complete <- drop_na(df_raw)
# 多重插补（使用mice，m=5，取第一个完整数据集）
set.seed(123)
mice_imp <- mice(df_filtered, m = 5, method = 'pmm', printFlag = FALSE)
df_imputed <- complete(mice_imp, 1)  # 选择第一个插补数据集

# 检查是否仍有缺失
stopifnot(all(!is.na(df_imputed)))

# # ---- 6. 相关性分析（使用插补后数据） ----
# # 将因子转换为数值（仅用于相关性计算）
# df_cor <- df_imputed %>%
#   mutate(across(where(is.factor), as.numeric))
# cor_matrix <- cor(df_cor, use = "pairwise.complete.obs", method = "spearman")
# diag(cor_matrix) <- 0
# 
# jpeg(file.path(output_dir, "correlation_heatmap_600dpi.jpg"),
#      width = 10, height = 8, units = "in", res = 600)
# corrplot(cor_matrix,
#          method = "color",
#          diag = FALSE,
#          type = "upper",
#          tl.cex = 0.8,
#          order = "AOE",
#          addCoef.col = "black",
#          col = COL2("RdBu"),
#          is.corr = FALSE,
#          col.lim = c(min(cor_matrix), max(cor_matrix)),
#          number.cex = 0.8,
#          title = "Spearman Correlation Matrix of All Variables",
#          mar = c(0,0,2,0)
# )
# dev.off()
# 
# # 保存相关性矩阵
# write.csv(cor_matrix, file.path(output_dir, "correlation_matrix.csv"))

# ---- 7. 数据划分（分层抽样，基于插补后数据） ----
set.seed(123)
split <- initial_split(df_imputed, prop = 0.80, strata = Pneumonia)  # 80%训练，20%测试
train_data <- training(split)
test_data  <- testing(split)

# ---- 8. LASSO特征选择（在训练集上进行） ----
# 准备LASSO所需格式（所有预测变量需数值化）
lasso_data <- train_data %>%
  mutate(across(where(is.factor), as.numeric)) %>%
  select(-Pneumonia)
lasso_x <- as.matrix(lasso_data)
lasso_y <- ifelse(train_data$Pneumonia == "yes", 1, 0)

# 交叉验证选择lambda
set.seed(456)
cv_lasso <- cv.glmnet(lasso_x, lasso_y, family = "binomial", alpha = 1, nfolds = 10)
best_lambda <- cv_lasso$lambda.min

# 提取非零系数特征
lasso_coef <- coef(cv_lasso, s = best_lambda)
selected_features <- rownames(lasso_coef)[which(lasso_coef[,1] != 0)][-1]  # 去掉截距
cat("LASSO selected features:", selected_features, "\n")

# 绘制LASSO交叉验证曲线并保存
jpeg(file.path(output_dir, "lasso_cv_curve_600dpi.jpg"),
     width = 8, height = 6, units = "in", res = 600)
plot(cv_lasso)
title("LASSO Regression Cross-Validation", line = 3)
dev.off()

# 绘制系数路径图
jpeg(file.path(output_dir, "lasso_path_600dpi.jpg"),
     width = 8, height = 6, units = "in", res = 600)
plot(cv_lasso$glmnet.fit, xvar = "lambda", label = TRUE)
abline(v = log(best_lambda), lty = 2, col = "red")
title("LASSO Coefficient Path", line = 3)
dev.off()

# ---- 8.1. 相关性分析（LASSO） ----
# 将因子转换为数值（仅用于相关性计算）
data_Lasso <- df_imputed[, c("Pneumonia", selected_features)]
df_cor <- data_Lasso %>%
  mutate(across(where(is.factor), as.numeric))
cor_matrix <- cor(df_cor, use = "pairwise.complete.obs", method = "spearman")
diag(cor_matrix) <- 0

jpeg(file.path(output_dir, "correlation_heatmap_lasso.jpg"),
     width = 10, height = 8, units = "in", res = 600)
corrplot(cor_matrix,
         method = "color",
         diag = FALSE,
         type = "upper",
         tl.cex = 0.8,
         order = "AOE",
         addCoef.col = "black",
         col = COL2("RdBu"),
         is.corr = FALSE,
         col.lim = c(min(cor_matrix), max(cor_matrix)),
         number.cex = 0.8,
         title = "Spearman Correlation Matrix of All Variables",
         mar = c(0,0,2,0)
)
dev.off()

# 保存相关性矩阵
write.csv(cor_matrix, file.path(output_dir, "correlation_matrix_LASSO.csv"))

# ---- 9. 根据LASSO筛选的变量更新数据 ----
# 保留Pneumonia和筛选出的特征
all_vars <- c("Pneumonia", selected_features)
train_selected <- train_data %>% select(any_of(all_vars))
test_selected  <- test_data %>% select(any_of(all_vars))

# ---- 10. 定义预处理方案（SMOTE，标准化，哑变量） ----
base_recipe <- recipe(Pneumonia ~ ., data = train_selected) %>%
  step_impute_median(all_numeric_predictors()) %>%   # 理论上已无缺失，但保留
  step_impute_mode(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE, keep_original_cols = FALSE) %>%
  step_zv(all_predictors()) %>%
  step_smote(Pneumonia, over_ratio = 1)

prep_recipe <- prep(base_recipe, training = train_selected)
train_preproc <- bake(prep_recipe, new_data = NULL)
test_preproc  <- bake(prep_recipe, new_data = test_selected)

# ---- 11. 定义8种模型规格（包含调参设定） ----
# 11.1 逻辑回归
log_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# 11.2 弹性网络（需调参）
en_spec <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# 11.3 随机森林
rf_spec <- rand_forest(trees = 1000, mtry = tune()) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# 11.4 XGBoost
xgb_spec <- boost_tree(trees = 1000, tree_depth = tune(), learn_rate = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# 11.5 支持向量机（RBF核）
svm_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# 11.6 K-最近邻
knn_spec <- nearest_neighbor(neighbors = tune()) %>%
  set_engine("kknn") %>%
  set_mode("classification")

# 11.7 决策树
dt_spec <- decision_tree(cost_complexity = tune(), tree_depth = tune()) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# 11.8 多层感知器（MLP）
mlp_spec <- mlp(hidden_units = tune(), penalty = tune(), epochs = 100) %>%
  set_engine("nnet") %>%
  set_mode("classification")

# ---- 12. 创建工作流集 ----
log_wf <- workflow() %>% add_recipe(base_recipe) %>% add_model(log_spec)
en_wf  <- workflow() %>% add_recipe(base_recipe) %>% add_model(en_spec)
rf_wf  <- workflow() %>% add_recipe(base_recipe) %>% add_model(rf_spec)
xgb_wf <- workflow() %>% add_recipe(base_recipe) %>% add_model(xgb_spec)
svm_wf <- workflow() %>% add_recipe(base_recipe) %>% add_model(svm_spec)
knn_wf <- workflow() %>% add_recipe(base_recipe) %>% add_model(knn_spec)
dt_wf  <- workflow() %>% add_recipe(base_recipe) %>% add_model(dt_spec)
mlp_wf <- workflow() %>% add_recipe(base_recipe) %>% add_model(mlp_spec)

wf_list <- list(
  Logistic = log_wf,
  EN = en_wf,
  RF = rf_wf,
  XGB = xgb_wf,
  SVM = svm_wf,
  KNN = knn_wf,
  DT = dt_wf,
  MLP = mlp_wf
)

# ---- 13. 交叉验证调参 ----
set.seed(789)
folds <- vfold_cv(train_selected, v = 10, strata = Pneumonia)

# 定义各模型的参数网格（简单示例，实际可更细致）
en_grid <- grid_regular(penalty(range = c(-3, 0)), mixture(range = c(0, 1)), levels = 4)
rf_grid <- grid_regular(mtry(range = c(2, 10)), levels = 5)
xgb_grid <- grid_regular(tree_depth(range = c(3, 10)), learn_rate(range = c(0.01, 0.3)), levels = 4)
svm_grid <- grid_regular(cost(range = c(-2, 2)), rbf_sigma(range = c(-3, 1)), levels = 4)
knn_grid <- grid_regular(neighbors(range = c(3, 21)), levels = 5)
dt_grid <- grid_regular(cost_complexity(range = c(-4, -1)), tree_depth(range = c(3, 15)), levels = 4)
mlp_grid <- grid_regular(hidden_units(range = c(3, 15)), penalty(range = c(-5, 0)), levels = 4)

# 执行调参（为简化，只对部分模型调参，其余保持默认）
tune_results <- list()
tune_results$EN <- tune_grid(en_wf, resamples = folds, grid = en_grid, metrics = metric_set(roc_auc))
tune_results$RF <- tune_grid(rf_wf, resamples = folds, grid = rf_grid, metrics = metric_set(roc_auc))
tune_results$XGB <- tune_grid(xgb_wf, resamples = folds, grid = xgb_grid, metrics = metric_set(roc_auc))
tune_results$SVM <- tune_grid(svm_wf, resamples = folds, grid = svm_grid, metrics = metric_set(roc_auc))
tune_results$KNN <- tune_grid(knn_wf, resamples = folds, grid = knn_grid, metrics = metric_set(roc_auc))
tune_results$DT <- tune_grid(dt_wf, resamples = folds, grid = dt_grid, metrics = metric_set(roc_auc))
tune_results$MLP <- tune_grid(mlp_wf, resamples = folds, grid = mlp_grid, metrics = metric_set(roc_auc))

# 提取最佳参数并更新工作流
best_params <- list()
for (model in names(tune_results)) {
  best <- select_best(tune_results[[model]], metric = "roc_auc")
  best_params[[model]] <- best
  wf_list[[model]] <- finalize_workflow(wf_list[[model]], best)
}

# 逻辑回归无需调参，直接使用

# ---- 14. 拟合所有模型 ----
fitted_models <- list()
for (model in names(wf_list)) {
  fitted_models[[model]] <- fit(wf_list[[model]], data = train_selected)
}

# ---- 15. 获取各模型预测概率 ----
get_prob <- function(model, newdata) {
  predict(model, newdata, type = "prob")$.pred_yes
}

train_probs <- map_dfc(fitted_models, ~get_prob(.x, train_selected)) %>%
  bind_cols(select(train_selected, Pneumonia))
test_probs <- map_dfc(fitted_models, ~get_prob(.x, test_selected)) %>%
  bind_cols(select(test_selected, Pneumonia))

# 保存预测概率
write.csv(test_probs, file.path(output_dir, "test_set_predictions.csv"), row.names = FALSE)

# ---- 16. 模型性能评估 ----
# 16.1 计算测试集AUC
model_auc <- sapply(names(fitted_models), function(m) {
  roc(test_selected$Pneumonia, test_probs[[m]], levels = c("no", "yes"))$auc
})
best_model <- names(which.max(model_auc))
cat("Best model based on test AUC:", best_model, "with AUC =", round(max(model_auc), 4), "\n")

# 16.2 绘制所有模型ROC曲线（600dpi）
roc_list <- list()
auc_values <- c()
for (model in names(fitted_models)) {
  roc_obj <- roc(test_selected$Pneumonia, test_probs[[model]], levels = c("no", "yes"))
  roc_list[[model]] <- roc_obj
  auc_values[model] <- round(auc(roc_obj), 3)
}
colors <- RColorBrewer::brewer.pal(8, "Set1")
names(colors) <- names(fitted_models)

jpeg(file.path(output_dir, "roc_curves_all_models_600dpi.jpg"),
     width = 10, height = 8, units = "in", res = 600)
plot(roc_list[[1]], col = colors[1], lwd = 2,
     main = "ROC Curves for All Models (Test Set)",
     xlab = "1 - Specificity", ylab = "Sensitivity",
     cex.main = 1.5, cex.lab = 1.2)
for (i in 2:length(roc_list)) {
  lines(roc_list[[i]], col = colors[i], lwd = 2)
}
legend("bottomright",
       legend = paste0(names(roc_list), " (AUC=", auc_values, ")"),
       col = colors, lwd = 2, cex = 0.9)
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()

# ---- 16.3 绘制平滑校准曲线（所有模型） ----
calibration_data <- test_probs %>%
  pivot_longer(cols = -Pneumonia, names_to = "Model", values_to = "pred_prob") %>%
  mutate(obs = as.numeric(Pneumonia == "yes"))  # 转换为 0/1

p_calib <- ggplot(calibration_data, aes(x = pred_prob, y = obs, color = Model)) +
  geom_smooth(method = "loess", se = FALSE, span = 0.8, size = 1) +  # LOESS 平滑
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50", size = 0.8) +
  scale_color_manual(values = colors) +                              # 使用之前定义的8种颜色
  labs(title = "Calibration Curves (Smooth, Test Set)",
       x = "Predicted Probability", y = "Observed Proportion") +
  theme_bw(base_size = 14) +
  theme(legend.position = "bottom") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))                    # 固定坐标轴范围
ggsave(file.path(output_dir, "calibration_curves_smooth_600dpi.jpg"),
       plot = p_calib, width = 10, height = 8, dpi = 600)

# ---- 16.4 决策曲线分析（DCA） ----
dca_data <- test_probs %>%
  mutate(Pneumonia_binary = ifelse(Pneumonia == "yes", 1, 0))
dca_formula <- as.formula(paste("Pneumonia_binary ~", paste(names(fitted_models), collapse = " + ")))
dca_results <- dca(dca_formula, data = dca_data, thresholds = seq(0, 1, by = 0.01))

# 获取实际曲线数量（模型 + all + none）
curve_names <- as.character(unique(dca_results$dca$label))
n_curves <- length(curve_names)
cat("决策曲线包含的线条：", paste(curve_names, collapse = ", "), "\n")

# 生成足够数量的颜色（使用扩展调色板）
if (n_curves <= 12) {
  colors_dca <- RColorBrewer::brewer.pal(n_curves, "Set3")  # Set3 最多12色
} else {
  colors_dca <- scales::hue_pal()(n_curves)                 # 自动生成任意数量
}

# 可选：定义线型（与颜色数量一致）
linetypes_dca <- rep(c("solid", "dashed", "dotted", "dotdash", "longdash", "twodash"), 
                     length.out = n_curves)

# 绘制决策曲线，限制横坐标 0~1
p_dca <- plot(dca_results, smooth = TRUE) +
  scale_color_manual(values = colors_dca) +
  scale_linetype_manual(values = linetypes_dca) +
  coord_cartesian(xlim = c(0, 0.3)) + coord_cartesian(ylim = c(-0.1,0.1)) +
  theme_bw(base_size = 14) +
  labs(title = "Decision Curve Analysis (Threshold 0–0.2)",
       x = "Threshold Probability", y = "Net Benefit") +
  theme(legend.position = "bottom",
        legend.title = element_blank())

ggsave(file.path(output_dir, "decision_curve_600dpi.jpg"),
       plot = p_dca, width = 10, height = 10, dpi = 600)


# 16.5 计算详细性能指标（准确率、灵敏度、特异度、F1等）
# ---- 16.6 计算各模型性能指标的95%置信区间 ----
library(pROC)
library(dplyr)
library(tidyr)

# 定义bootstrap函数计算F1的置信区间
boot_f1_ci <- function(truth, pred_class, n_boot = 1000, conf_level = 0.95) {
  # truth: 真实类别（因子，水平与预测一致）
  # pred_class: 预测类别（因子）
  n <- length(truth)
  f1_values <- numeric(n_boot)
  set.seed(123)  # 可重复
  for (i in 1:n_boot) {
    idx <- sample(n, replace = TRUE)
    truth_boot <- truth[idx]
    pred_boot <- pred_class[idx]
    cm <- table(truth_boot, pred_boot)
    # 处理可能缺失的类别水平
    if (ncol(cm) == 1 || nrow(cm) == 1) {
      # 若所有预测均为同一类，则F1可能为0或1，这里简单处理为NA
      f1_values[i] <- NA
      next
    }
    tp <- cm[2, 2]  # 假设第二行/列是"yes"
    fp <- cm[1, 2]
    fn <- cm[2, 1]
    precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
    recall    <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
    f1 <- ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
    f1_values[i] <- f1
  }
  f1_values <- f1_values[!is.na(f1_values)]
  if (length(f1_values) < n_boot/2) {
    warning("Bootstrap samples with valid F1 are insufficient.")
    return(c(NA, NA))
  }
  alpha <- 1 - conf_level
  ci <- quantile(f1_values, probs = c(alpha/2, 1 - alpha/2), na.rm = TRUE)
  return(ci)
}

# 初始化结果列表
results_list <- list()

for (model in names(fitted_models)) {
  cat("Calculating confidence intervals for", model, "...\n")
  
  # 获取真实值和预测概率、预测类别
  truth <- test_selected$Pneumonia  # 因子，水平为 "no", "yes"
  pred_class <- predict(fitted_models[[model]], test_selected, type = "class")$.pred_class
  prob <- test_probs[[model]]
  
  # 确保因子水平一致（"yes" 为阳性）
  truth <- factor(truth, levels = c("no", "yes"))
  pred_class <- factor(pred_class, levels = c("no", "yes"))
  
  # 混淆矩阵
  cm <- table(truth, pred_class)
  tp <- cm["yes", "yes"]
  tn <- cm["no", "no"]
  fp <- cm["no", "yes"]
  fn <- cm["yes", "no"]
  n_total <- sum(cm)
  n_pos <- tp + fn
  n_neg <- tn + fp
  
  # ---- AUC 及其 CI (DeLong法) ----
  roc_obj <- roc(truth, prob, levels = c("no", "yes"), quiet = TRUE)
  auc_val <- auc(roc_obj)
  auc_ci <- ci.auc(roc_obj, method = "delong")  # 返回长度为3的向量：下限、AUC、上限
  auc_lower <- auc_ci[1]
  auc_upper <- auc_ci[3]
  
  # ---- Accuracy CI (二项精确法) ----
  acc_val <- (tp + tn) / n_total
  acc_ci <- binom.test(tp + tn, n_total)$conf.int  # 默认0.95
  
  # ---- Sensitivity CI (二项精确法) ----
  sens_val <- tp / n_pos
  sens_ci <- binom.test(tp, n_pos)$conf.int
  
  # ---- Specificity CI (二项精确法) ----
  spec_val <- tn / n_neg
  spec_ci <- binom.test(tn, n_neg)$conf.int
  
  # ---- F1 及其 CI (bootstrap) ----
  # 使用之前定义的函数
  f1_val <- 2 * tp / (2 * tp + fp + fn)  # 直接计算F1
  f1_ci <- boot_f1_ci(truth, pred_class, n_boot = 1000)
  
  # 将结果整理为一行
  results_list[[model]] <- data.frame(
    Model = model,
    AUC = round(auc_val, 3),
    AUC_lower = round(auc_lower, 3),
    AUC_upper = round(auc_upper, 3),
    Accuracy = round(acc_val, 3),
    Acc_lower = round(acc_ci[1], 3),
    Acc_upper = round(acc_ci[2], 3),
    Sensitivity = round(sens_val, 3),
    Sens_lower = round(sens_ci[1], 3),
    Sens_upper = round(sens_ci[2], 3),
    Specificity = round(spec_val, 3),
    Spec_lower = round(spec_ci[1], 3),
    Spec_upper = round(spec_ci[2], 3),
    F1 = round(f1_val, 3),
    F1_lower = round(f1_ci[1], 3),
    F1_upper = round(f1_ci[2], 3)
  )
}

# 合并所有结果
all_metrics_ci <- bind_rows(results_list)

# 可选：生成一个合并了估计值和CI的友好格式表格
metrics_pretty <- all_metrics_ci %>%
  mutate(
    AUC_CI = paste0(AUC, " (", AUC_lower, "-", AUC_upper, ")"),
    Accuracy_CI = paste0(Accuracy, " (", Acc_lower, "-", Acc_upper, ")"),
    Sensitivity_CI = paste0(Sensitivity, " (", Sens_lower, "-", Sens_upper, ")"),
    Specificity_CI = paste0(Specificity, " (", Spec_lower, "-", Spec_upper, ")"),
    F1_CI = paste0(F1, " (", F1_lower, "-", F1_upper, ")")
  ) %>%
  select(Model, AUC_CI, Accuracy_CI, Sensitivity_CI, Specificity_CI, F1_CI)

# 保存完整表格（含上下限列）
write.csv(all_metrics_ci, file.path(output_dir, "all_models_performance_with_CI.csv"), row.names = FALSE)
# 保存简洁格式表格
write.csv(metrics_pretty, file.path(output_dir, "all_models_performance_pretty.csv"), row.names = FALSE)

# 打印到控制台
print(metrics_pretty)

# ---- 17. 对 XGBoost 模型进行细致的 SHAP 分析 ----
# 检查 XGBoost 模型是否存在
if ("XGB" %in% names(fitted_models)) {
  cat("\nPerforming SHAP analysis for XGBoost ...\n")
  
  # 提取 XGBoost 模型
  xgb_fit <- fitted_models[["XGB"]]
  xgb_engine <- extract_fit_engine(xgb_fit)  # xgb.Booster 对象
  model_type <- "XGBoost"
  
  # 准备特征数据（使用预处理后的训练集和测试集）
  feature_names <- setdiff(names(train_preproc), "Pneumonia")
  X_train <- train_preproc[, feature_names, drop = FALSE]
  X_test  <- test_preproc[, feature_names, drop = FALSE]
  
  # 确保所有特征为数值型（shapviz 要求）
  X_train <- as.data.frame(lapply(X_train, as.numeric))
  X_test  <- as.data.frame(lapply(X_test, as.numeric))
  
  my_theme <- theme(
    text = element_text(size = 16),                # 文字大小
    plot.title = element_text(size = 18, face = "bold"),
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 15),
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 13),
    #legend.key.height = unit(1, "npc"),       # 使图例与 panel 等高
    panel.background = element_rect(fill = "white"),   # 绘图面板背景设为白色
    plot.background = element_rect(fill = "white"),    # 整个图形背景设为白色（可选）
    panel.grid.major = element_blank(),                # 去掉主要网格线（可选）
    panel.grid.minor = element_blank(),                 # 去掉次要网格线（可选）
    panel.border = element_rect(fill = NA, linewidth = 0.5)
  )
  # 创建 shapviz 对象（训练集和测试集）
  shp_train <- shapviz(xgb_engine, X_pred = as.matrix(X_train), X = X_train)
  
  num_feature <- length(selected_features)
  # ---- 图1：SHAP 重要性条形图（基于训练集） ----
  # 显示前10个最重要特征（删除低贡献变量）
  p1 <- sv_importance(shp_train, kind = "bar", fill = "steelblue",
                      max_display = num_feature) +
                      my_theme
  ggsave(file.path(output_dir, paste0("shap_importance_XGB_600dpi.jpg")),
         plot = p1, width = 10, height = 8, dpi = 600)
  
  # ---- 图2：SHAP 蜂群图（基于训练集） ----
  p2 <- sv_importance(shp_train, 
                      kind = "beeswarm", 
                      max_display = num_feature,
                      color = c(low = "black",  high = "red"), 
                      bee_width = 0.2,
                      show_numbers = TRUE,
                      size = 1) + 
                      my_theme +
                      theme(legend.key.height = unit(1, "npc"))
  
  ggsave(file.path(output_dir, paste0("shap_beeswarm_XGB_600dpi.jpg")),
         plot = p2, width = 10, height = 8, dpi = 600)
  
  # ---- 图3：瀑布图（基于测试集第一个阳性样本） ----
  pos_sample <- which(test_selected$Pneumonia == "yes")[1]
  if (!is.na(pos_sample)) {
    shp_test <- shapviz(xgb_engine, X_pred = as.matrix(X_test), X = X_test)
    p3 <- sv_waterfall(shp_test, row_id = pos_sample) +
      labs(title = paste0("Waterfall Plot for Test Sample ", pos_sample,
                          "\nTrue: ", test_selected$Pneumonia[pos_sample], " - ", model_type)) + my_theme
    ggsave(file.path(output_dir, paste0("shap_waterfall_XGB_600dpi.jpg")),
           plot = p3, width = 8, height = 6, dpi = 600)
  }
  
  # ---- 输出特征重要性表格（基于训练集） ----
  imp <- data.frame(
    Feature = names(colMeans(abs(shp_train$S))),
    Importance = colMeans(abs(shp_train$S))
  ) %>% arrange(desc(Importance))
  write.csv(imp, file.path(output_dir, "shap_importance_XGB.csv"), row.names = FALSE)
  
  cat("SHAP analysis completed for XGBoost.\n")
} else {
  warning("XGBoost model not found in fitted_models. Skipping SHAP analysis.")
}


# ---- 18. 逻辑回归显著变量输出（同原代码） ----
if ("Logistic" %in% names(fitted_models)) {
  log_glm <- fitted_models$Logistic %>% extract_fit_parsnip() %>% pluck("fit")
  coef_summary <- summary(log_glm)$coefficients
  coef_df <- data.frame(
    Variable = rownames(coef_summary),
    Estimate = coef_summary[, "Estimate"],
    StdError = coef_summary[, "Std. Error"],
    z_value = coef_summary[, "z value"],
    p_value = coef_summary[, "Pr(>|z|)"]
  ) %>%
    filter(Variable != "(Intercept)") %>%
    mutate(Significance = case_when(
      p_value < 0.001 ~ "***",
      p_value < 0.01  ~ "**",
      p_value < 0.05  ~ "*",
      TRUE            ~ "ns"
    )) %>%
    arrange(p_value)
  significant_vars <- coef_df %>% filter(p_value < 0.05)
  write.csv(significant_vars, file.path(output_dir, "significant_variables_logistic.csv"), row.names = FALSE)
}

# ---- 19. 保存所有模型 ----
saveRDS(fitted_models, file.path(output_dir, "all_fitted_models.rds"))

# ---- 20. 生成基线特征表（Table 1 风格） ----
table1 <- df_imputed %>%
  group_by(Pneumonia) %>%
  summarise(across(where(is.numeric), list(mean = ~mean(., na.rm = TRUE), sd = ~sd(., na.rm = TRUE))),
            across(where(is.factor), list(count = ~n(), prop = ~mean(. == "yes"))))
write.csv(table1, file.path(output_dir, "baseline_characteristics.csv"), row.names = FALSE)

cat("\n========== ALL DONE ==========\n")
cat("Results saved in folder:", output_dir, "\n")
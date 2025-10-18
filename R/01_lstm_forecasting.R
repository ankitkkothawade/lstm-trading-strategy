# -----------------------------------------------------------
# 01_lstm_forecasting.R
# LSTM price forecasting + simple trading for 5 stocks
# Saves per-stock charts & a combined results CSV to /results
# -----------------------------------------------------------

suppressPackageStartupMessages({
  library(quantmod)
  library(dplyr)
  library(ggplot2)
  library(scales)
  library(PerformanceAnalytics)
  library(keras)
  library(tidyr)
})

# ----- Safety: check TensorFlow/keras backend -----
if (!keras::is_keras_available()) {
  message("\n❗ Keras/TensorFlow backend not found.\n",
          "Run this once in R:\n",
          "  install.packages('keras')\n",
          "  library(keras); install_keras()\n",
          "Then re-run this script.\n")
  quit(save = "no")
}

# ----- Params -----
symbols       <- c("MSFT","AAPL","GOOG","WMT","HPE")
train_start   <- as.Date("2013-01-01")
train_end     <- as.Date("2020-12-31")
test_end      <- as.Date("2022-12-31")
capital_each  <- 2000     # £ per stock
thr           <- 0.025    # +/-2.5% threshold for buy/sell signals
lookback      <- 30       # sequence length
epochs_       <- 15
batch_        <- 32

# ----- Folders -----
if (!dir.exists("results")) dir.create("results", recursive = TRUE)

# ----- Helpers -----
scale_minmax   <- function(x, a, b) (x - a) / (b - a)
inv_scale_mm   <- function(x, a, b)  x * (b - a) + a

make_sequences <- function(v, L = 30) {
  n <- length(v)
  if (n <= L) stop("Not enough observations to create sequences.")
  X <- array(0, dim = c(n - L, L, 1))
  y <- array(0, dim = c(n - L))
  for (i in seq_len(n - L)) {
    X[i,,1] <- v[i:(i + L - 1)]
    y[i]    <- v[i + L]
  }
  list(X = X, y = y)
}

run_one_symbol <- function(symbol) {
  # ----- Data -----
  suppressWarnings(getSymbols(symbol, from = train_start, to = test_end, src = "yahoo", auto.assign = TRUE))
  px <- get(symbol)
  close <- Cl(px)
  df <- data.frame(date = as.Date(index(close)), close = as.numeric(close))

  # Split
  df_train <- df %>% filter(date <= train_end)
  df_test  <- df %>% filter(date >  train_end & date <= test_end)
  if (nrow(df_train) < lookback + 10 || nrow(df_test) < lookback + 10) {
    stop(paste0("Insufficient data for ", symbol, " after splits."))
  }

  # Scale to train min/max (prevents leakage)
  min_c <- min(df_train$close); max_c <- max(df_train$close)
  df$close_s <- scale_minmax(df$close, min_c, max_c)

  # Sequences
  v_train <- df$close_s[df$date <= train_end]
  v_test  <- df$close_s[df$date >  train_end & df$date <= test_end]

  # Test sequences are built contiguous after train tail
  seq_train <- make_sequences(v_train, lookback)
  seq_test  <- make_sequences(c(tail(v_train, lookback), v_test), lookback)

  X_train <- seq_train$X; y_train <- seq_train$y
  X_test  <- seq_test$X;  y_test  <- seq_test$y

  # ----- Model -----
  keras::k_clear_session()
  model <- keras_model_sequential() |>
    layer_lstm(units = 50, input_shape = c(lookback, 1), return_sequences = FALSE) |>
    layer_dense(units = 1)

  model |>
    compile(optimizer = "adam", loss = "mse")

  invisible(model |>
              fit(X_train, y_train,
                  epochs = epochs_, batch_size = batch_,
                  validation_split = 0.1, verbose = 0))

  # Predict on test
  pred_s <- as.numeric(model %>% predict(X_test, verbose = 0))
  true_s <- as.numeric(y_test)

  # Inverse scale to price
  pred <- inv_scale_mm(pred_s, min_c, max_c)
  true <- inv_scale_mm(true_s, min_c, max_c)

  # Align dates for test predictions
  test_dates <- df %>% filter(date > train_end & date <= test_end) %>% pull(date)

  # y_test aligns with the first test day because we prefixed with tail(v_train, lookback)
  pred_dates <- test_dates

  # Force equal lengths (belt-and-braces)
  n <- min(length(pred), length(true), length(pred_dates))
  pred <- pred[seq_len(n)]
  true <- true[seq_len(n)]
  pred_dates <- pred_dates[seq_len(n)]

  # Build frame and compute returns with LAG (no c(NA, ...))
  pred_df <- data.frame(date = pred_dates, pred = pred, true = true)

  lag_true <- dplyr::lag(pred_df$true)
  pred_df$ret_true <- (pred_df$true / lag_true) - 1
  pred_df$ret_pred <- (pred_df$pred / lag_true) - 1

  # Signals from predicted return vs last true
  pred_df$signal <- dplyr::case_when(
    pred_df$ret_pred >=  thr ~  1L,   # BUY
    pred_df$ret_pred <= -thr ~ -1L,   # SELL (we go flat)
    TRUE                     ~  0L
  )

  # Long-only, hold between signals
  pred_df$position <- 0L
  pos <- 0L
  for (i in seq_len(nrow(pred_df))) {
    sig <- pred_df$signal[i]
    if (sig == 1L) pos <- 1L else if (sig == -1L) pos <- 0L
    pred_df$position[i] <- pos
  }

  # Strategy returns
  pred_df$strat_ret <- pred_df$position * pred_df$ret_true
  pred_df$strat_ret[is.na(pred_df$strat_ret)] <- 0

  # Equity curve
  equity <- cumprod(1 + pred_df$strat_ret) * capital_each
  pred_df$equity <- equity

  # Metrics
  daily_ret <- na.omit(pred_df$strat_ret)
  ann_ret  <- if (length(daily_ret) > 1) mean(daily_ret) * 252 else NA_real_
  ann_vol  <- if (length(daily_ret) > 1) sd(daily_ret) * sqrt(252) else NA_real_
  sharpe   <- ifelse(!is.na(ann_vol) && ann_vol > 0, ann_ret / ann_vol, NA_real_)
  profit   <- tail(pred_df$equity, 1) - capital_each
  acc      <- mean(sign(pred_df$ret_true) == sign(pred_df$ret_pred), na.rm = TRUE)

  # ----- Save charts -----
  p1 <- ggplot(pred_df, aes(date)) +
    geom_line(aes(y = true), linewidth = 0.7, alpha = 0.9) +
    geom_line(aes(y = pred), linewidth = 0.7, alpha = 0.9, linetype = "dashed") +
    scale_y_continuous(labels = dollar_format(prefix = "$")) +
    labs(title = paste0(symbol, " — True vs Predicted Close (Test)"),
         x = NULL, y = "Price") +
    theme_minimal()
  ggsave(paste0("results/closing_prices_", symbol, ".png"), p1, width = 9, height = 5, dpi = 150)

  p2 <- ggplot(pred_df, aes(date, equity)) +
    geom_line(linewidth = 0.8, color = "steelblue") +
    labs(title = paste0(symbol, " — LSTM Strategy Equity Curve"),
         x = NULL, y = paste0("Equity (base = £", capital_each, ")")) +
    theme_minimal()
  ggsave(paste0("results/lstm_equity_curve_", symbol, ".png"), p2, width = 9, height = 5, dpi = 150)

  # Return per-stock summary row
  data.frame(
    symbol = symbol,
    final_value = round(tail(equity, 1), 2),
    profit = round(profit, 2),
    sharpe = round(sharpe, 3),
    directional_accuracy = round(acc * 100, 1),
    stringsAsFactors = FALSE
  )
}

# ----- Run for all symbols -----
results <- do.call(rbind, lapply(symbols, run_one_symbol))

# Combined portfolio (equal capital per stock)
total_initial <- capital_each * length(symbols)
total_final   <- sum(results$final_value)
portfolio_profit <- round(total_final - total_initial, 2)

# Save table
write.csv(results, "results/final_values.csv", row.names = FALSE)

cat("\n✅ LSTM runs complete for: ", paste(symbols, collapse = ", "), "\n",
    "Total initial (£): ", total_initial, "\n",
    "Total final (£):   ", round(total_final, 2), "\n",
    "Portfolio profit:  ", portfolio_profit, "\n",
    "\nPer-stock summary saved to results/final_values.csv\n",
    "Charts saved per stock in /results\n")

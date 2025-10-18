# -----------------------------------------------------------
# 02_strategy_comparison.R
# Baselines vs LSTM: Buy&Hold, Always-Buy, Mean Predictor, Random
# Saves per-stock table and portfolio totals + comparison plots
# -----------------------------------------------------------

suppressPackageStartupMessages({
  library(quantmod)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
  library(PerformanceAnalytics)
  library(TTR)
})

# ---------- Params (match your LSTM script) ----------
symbols      <- c("MSFT","AAPL","GOOG","WMT","HPE")
train_start  <- as.Date("2013-01-01")
train_end    <- as.Date("2020-12-31")
test_end     <- as.Date("2022-12-31")
capital_each <- 2000
thr          <- 0.025   # +/- 2.5% buy/sell threshold for signal-based strategies
mean_k       <- 5       # rolling window for mean predictor

if (!dir.exists("results")) dir.create("results", recursive = TRUE)

# ---------- Helper: fetch test prices & returns ----------
get_test_df <- function(symbol) {
  suppressWarnings(getSymbols(symbol, from = train_start, to = test_end, src = "yahoo", auto.assign = TRUE))
  px <- get(symbol)
  close <- Cl(px)
  df <- data.frame(date = as.Date(index(close)), close = as.numeric(close))
  df_test <- df %>% filter(date > train_end & date <= test_end)
  df_test <- df_test %>% mutate(ret_true = close / dplyr::lag(close) - 1)
  df_test
}

# ---------- Strategies ----------
buy_hold_equity <- function(df_test, capital) {
  first_px <- df_test$close[1]
  eq <- capital * (df_test$close / first_px)
  eq
}

always_buy_equity <- function(df_test, capital) {
  # identical to buy & hold for long-only
  buy_hold_equity(df_test, capital)
}

mean_predictor_equity <- function(df_test, capital, k = 5, thr = 0.025) {
  # Predict next close by SMA of past k closes (no lookahead: SMA of lagged close)
  pred <- SMA(dplyr::lag(df_test$close), n = k)
  # predicted return vs last true close
  ret_pred <- (pred / dplyr::lag(df_test$close)) - 1

  # Signals
  sig <- dplyr::case_when(
    ret_pred >=  thr ~  1L,  # BUY
    ret_pred <= -thr ~ -1L,  # SELL -> flat
    TRUE              ~  0L
  )

  # Position: long-only; hold until SELL
  pos <- integer(nrow(df_test)); cur <- 0L
  for (i in seq_len(nrow(df_test))) {
    s <- sig[i]
    if (!is.na(s)) {
      if (s == 1L) cur <- 1L else if (s == -1L) cur <- 0L
    }
    pos[i] <- cur
  }

  strat_ret <- pos * df_test$ret_true
  strat_ret[is.na(strat_ret)] <- 0
  eq <- cumprod(1 + strat_ret) * capital
  eq
}

random_equity <- function(df_test, capital, p = 0.5) {
  set.seed(42)
  pos <- rbinom(nrow(df_test), size = 1, prob = p)
  strat_ret <- pos * df_test$ret_true
  strat_ret[is.na(strat_ret)] <- 0
  eq <- cumprod(1 + strat_ret) * capital
  eq
}

# ---------- Run per stock ----------
per_stock <- list()

for (sym in symbols) {
  df_test <- get_test_df(sym)

  # Guard if too few rows
  if (nrow(df_test) < 50) {
    warning("Too few test rows for ", sym, " — skipping.")
    next
  }

  eq_bh   <- buy_hold_equity(df_test, capital_each)
  eq_alw  <- always_buy_equity(df_test, capital_each)
  eq_mean <- mean_predictor_equity(df_test, capital_each, k = mean_k, thr = thr)
  eq_rand <- random_equity(df_test, capital_each, p = 0.5)

  # Final values
  row <- data.frame(
    symbol      = sym,
    BuyHold     = round(tail(eq_bh,   1), 2),
    AlwaysBuy   = round(tail(eq_alw,  1), 2),
    MeanPred    = round(tail(eq_mean, 1), 2),
    Random      = round(tail(eq_rand, 1), 2),
    stringsAsFactors = FALSE
  )
  per_stock[[length(per_stock) + 1]] <- row
}

baselines_df <- do.call(rbind, per_stock)

# ---------- Join with LSTM results ----------
lstm_path <- "results/final_values.csv"
if (!file.exists(lstm_path)) {
  stop("Missing 'results/final_values.csv' from 01_lstm_forecasting.R. Run that script first.")
}
lstm_df <- read.csv(lstm_path, stringsAsFactors = FALSE) %>%
  select(symbol, LSTM = final_value)

comparison <- baselines_df %>%
  left_join(lstm_df, by = "symbol") %>%
  relocate(symbol, LSTM, BuyHold, AlwaysBuy, MeanPred, Random)

# Save per-stock comparison
write.csv(comparison, "results/strategy_comparison_by_stock.csv", row.names = FALSE)

# ---------- Portfolio totals (sum across 5 stocks) ----------
totals <- data.frame(
  Strategy = c("LSTM", "BuyHold", "AlwaysBuy", "MeanPred", "Random"),
  Final_Value = c(
    sum(comparison$LSTM,    na.rm = TRUE),
    sum(comparison$BuyHold, na.rm = TRUE),
    sum(comparison$AlwaysBuy, na.rm = TRUE),
    sum(comparison$MeanPred, na.rm = TRUE),
    sum(comparison$Random,   na.rm = TRUE)
  ),
  stringsAsFactors = FALSE
)

totals$Initial_Value <- capital_each * length(symbols)
totals$Profit <- round(totals$Final_Value - totals$Initial_Value, 2)

write.csv(totals, "results/strategy_portfolio_totals.csv", row.names = FALSE)

# ---------- Plots ----------
# 1) Per-stock: bar chart of final values by strategy (long format)
long_stock <- comparison %>%
  pivot_longer(cols = -symbol, names_to = "Strategy", values_to = "Final_Value")

p1 <- ggplot(long_stock, aes(Strategy, Final_Value, fill = Strategy)) +
  geom_col(position = "dodge") +
  facet_wrap(~ symbol, scales = "free_y") +
  scale_y_continuous(labels = dollar_format(prefix = "£")) +
  labs(title = "Per-Stock Final Values by Strategy",
       x = NULL, y = "Final Value (£)") +
  theme_minimal() +
  theme(legend.position = "none")
ggsave("results/strategy_by_stock.png", p1, width = 11, height = 6, dpi = 150)

# 2) Portfolio totals: bar chart
p2 <- ggplot(totals, aes(Strategy, Final_Value, fill = Strategy)) +
  geom_col() +
  scale_y_continuous(labels = dollar_format(prefix = "£")) +
  labs(title = "Portfolio Final Value by Strategy (5 stocks, £2k each)",
       x = NULL, y = "Final Value (£)") +
  theme_minimal() +
  theme(legend.position = "none")
ggsave("results/strategy_comparison.png", p2, width = 9, height = 5, dpi = 150)

cat("\n✅ Strategy comparison complete.\n",
    "- Saved per-stock table: results/strategy_comparison_by_stock.csv\n",
    "- Saved portfolio totals: results/strategy_portfolio_totals.csv\n",
    "- Saved plots: results/strategy_by_stock.png, results/strategy_comparison.png\n")

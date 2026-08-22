"""The starter config written by ``ainvest init``."""

EXAMPLE_CONFIG = '''# AInvestment Planner configuration
#
# Everything the tool does is driven by this file. Edit it, re-run
# `ainvest optimize`, see what changes.
#
# IMPORTANT: the fee numbers below are ESTIMATES chosen to be realistic, not
# quoted rates. Fee schedules change and vary by account, order type and
# volume. Look up what YOUR venue charges YOU and put the real numbers here --
# for a few hundred dollars, fees matter more than anything else in this file.

budget: 300            # total dollars you are putting to work
cash_reserve: 0        # held back from the plan entirely
risk_profile: balanced # conservative | balanced | aggressive | degen

# How far a position may drift before a rebalance is worth doing, and the
# smallest trade worth placing at all.
rebalance_band: 0.05
min_trade: 5
allow_sells: false     # false = steer with new cash only; no taxable sales


venues:
  public:
    # Commission-free equities with fractional shares.
    fee_bps: 0
    spread_bps: 0
    min_notional: 1
    fractional: true
    size_decimals: 5

  public_crypto:
    # Retail crypto is usually sold with a markup baked into the price rather
    # than a visible commission. This models that markup.
    fee_bps: 0
    spread_bps: 175
    min_notional: 1
    fractional: true
    size_decimals: 8
    supports_withdrawal: false   # exposure without withdrawal is not self-custody

  coinbase_simple:
    # The one-click buy flow: convenient, and the most expensive way to buy.
    fee_bps: 0
    spread_bps: 150
    min_notional: 2
    fractional: true
    size_decimals: 8
    supports_withdrawal: true
    withdrawal_fee: 1.50

  coinbase_advanced:
    # Advanced Trade / limit orders. Materially cheaper than the simple flow;
    # if you are buying crypto at all, buy it here.
    fee_bps: 60
    spread_bps: 5
    min_notional: 2
    fractional: true
    size_decimals: 8
    supports_withdrawal: true
    withdrawal_fee: 1.50

  kraken_pro:
    fee_bps: 40
    spread_bps: 5
    min_notional: 5
    fractional: true
    size_decimals: 8
    supports_withdrawal: true
    withdrawal_fee: 1.00


assets:
  # --- Equities, bought on Public -----------------------------------------
  - symbol: VTI
    name: Total US stock market
    asset_class: equity
    venues: [public]
    max_weight: 0.60
    # a similar-but-not-identical fund to hold through a wash-sale window
    # after harvesting a loss here
    tlh_replacement: VOO

  - symbol: VXUS
    name: Total international stock market
    asset_class: equity
    venues: [public]
    max_weight: 0.40

  - symbol: SCHD
    name: US dividend equities
    asset_class: equity
    venues: [public]
    max_weight: 0.35

  - symbol: GLD
    name: Gold
    asset_class: equity     # not a stock, but it trades like one on Public
    venues: [public]
    max_weight: 0.25

  # --- Crypto --------------------------------------------------------------
  # data_symbol is what the price provider calls it; symbol is what you call it.
  - symbol: BTC
    name: Bitcoin
    asset_class: crypto
    data_symbol: BTC-USD
    venues: [coinbase_advanced, kraken_pro, coinbase_simple]
    max_weight: 0.20
    self_custody: true      # Trezor holds this natively

  - symbol: ETH
    name: Ethereum
    asset_class: crypto
    data_symbol: ETH-USD
    venues: [coinbase_advanced, kraken_pro, coinbase_simple]
    max_weight: 0.15
    self_custody: true


constraints:
  # Per-asset caps also come from each asset's own max_weight; the binding
  # one wins. Keep the caps loose enough to leave the optimiser room -- if
  # they barely sum to 1 there is only one feasible portfolio and every
  # objective returns it.
  single_max: 0.35        # no single position above this
  crypto_max: 0.25        # total crypto ceiling (the risk profile sets a default)
  crypto_min: 0.0
  include_threshold: 0.02 # weights under 2% get dropped rather than bought as dust
  max_positions: 0        # 0 = no limit


optimizer:
  # risk_parity uses only the covariance matrix, never the return estimates.
  # That makes it far more stable on a short history than max_sharpe, which is
  # why it is the default. Try the others and compare with `ainvest backtest`.
  objective: risk_parity      # risk_parity | max_sharpe | min_vol | target_vol | equal_weight | max_return
  risk_free_rate: 0.04
  mean_shrinkage: 0.5         # 0 = trust the sample means, 1 = ignore them
  covariance_shrinkage: ledoit_wolf
  blend_equal_weight: 0.20    # pull the answer 20% toward equal weight
  target_vol: 0.15            # only used by the target_vol objective


data:
  provider: yahoo         # yahoo | csv | synthetic
  lookback_days: 730
  cache_dir: .cache/prices
  cache_ttl_hours: 12
  min_observations: 60


custody:
  enabled: true
  wallet: Trezor
  sweep_threshold: 100    # don't withdraw a position smaller than this
  max_fee_fraction: 0.02  # or one where the network fee exceeds 2% of it


# The trade-worthiness gate: a trade's fee must be earned back within this
# many months by the expected return on the amount traded, or it is skipped.
fee_payback_months: 6


contribution:
  # The steady-investment engine. `ainvest check` plans where each
  # contribution goes -- always toward whatever is furthest below target.
  amount: 100
  frequency: monthly      # monthly | biweekly | weekly
  reserve_fraction: 0.0   # e.g. 0.25 holds a quarter back as dry powder...
  deploy_on_drawdown: 0.15  # ...flagged for deployment on a 15% drawdown


tax:
  # Marginal rates, used to estimate what selling costs and what a harvested
  # loss saves. Set both to 0 for a tax-sheltered account. These are YOUR
  # brackets -- look them up.
  short_term_rate: 0.24
  long_term_rate: 0.15
  harvest_min_loss: 25    # ignore losses smaller than this
  harvest_min_saving: 5   # ...or saving less tax than this
  wash_sale_days: 30
  # US law applies the wash-sale rule to "securities", which so far has not
  # included crypto -- but that is a recurring legislative target. Verify.
  crypto_wash_sale: false
  long_term_boundary_days: 45  # warn before selling a gain this close to long-term


signals:
  # Alerts computed by `ainvest check`. They never place or suggest sells on
  # their own; run `ainvest backtest --exit-rule trend` to see what acting on
  # the trend signal would actually have cost after tax.
  trend_days: 200
  drawdown_alert: 0.20


# What you already own, AS TAX LOTS: one entry per purchase, not one per
# symbol. cost_basis is the total paid for that lot (fees included);
# acquired is the purchase date. Both are what make tax analysis possible.
holdings: []
# holdings:
#   - symbol: VTI
#     quantity: 0.42
#     venue: public
#     cost_basis: 118.50        # total dollars paid for this lot
#     acquired: 2025-11-03
#   - symbol: BTC
#     quantity: 0.0015
#     venue: coinbase_advanced
#     cost_basis: 97.20
#     acquired: 2026-02-14
'''

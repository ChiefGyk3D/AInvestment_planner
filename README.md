# AInvestment Planner

A fee-aware allocation planner for **small** stock and crypto accounts — the
few-hundred-dollar kind, where transaction costs and order minimums matter more
than anything an optimizer can do.

It produces a **plan**: target weights, and the specific orders to place at each
venue to reach them. It does not connect to your brokerage, hold API keys, or
place trades. You read the plan and decide.

> **Not financial advice.** This is an educational modelling tool. Every number
> it prints is an estimate derived from past prices, and the past does not
> predict the future. You can lose money, including all of it. Verify every fee,
> minimum, and ticker against your own venue before placing an order.

---

## Why this exists

Most portfolio optimizers assume the transaction costs round to zero. At $300
they don't:

- A 2% weight of a $300 budget is a **$6 order**. If the venue takes $1 to place
  it, that position is down 17% before the market opens.
- Buying crypto through a one-click retail flow can cost **1.5–2%** in spread,
  invisibly, versus a fraction of that on the same exchange's advanced order
  screen. On $50 of BTC that difference is most of a year's expected return.
- Sweeping $30 of BTC to a hardware wallet across a $2 network fee costs **7%**
  of the position.

So this tool optimizes weights *and* then does the unglamorous part: drops
orders too small to be worth their fee, redistributes that money into what's
left, quantizes to what the venue will accept, and tells you what the whole
thing cost.

## Install

```bash
git clone https://github.com/ChiefGyk3D/AInvestment_planner
cd AInvestment_planner
pip install -e .            # core: numpy, pandas, pyyaml
pip install -e '.[live]'    # adds yfinance for real market data
```

Python 3.10 or newer.

## Quick start

```bash
ainvest init            # writes portfolio.yaml — edit it, especially the fees
ainvest explain         # what the data says: returns, vols, correlations
ainvest optimize        # the allocation and the orders to place
ainvest backtest        # sanity-check the method on past data
```

No network? Set `data.provider: synthetic` in the config and everything runs on
a deterministic fake market. Useful for trying the tool out, and it's what the
test suite uses.

### What `optimize` prints

```
Target allocation (risk_parity)
  asset  weight  exp. return  volatility  risk share
  VTI     20.3%        12.3%       16.8%       15.4%
  VXUS    19.0%        10.8%       18.0%       15.6%
  SCHD    23.1%        11.3%       14.8%       15.3%
  GLD     23.3%         9.8%       14.6%        7.6%
  BTC      7.5%        25.9%       65.4%       21.9%
  ETH      6.8%        23.5%       82.9%       24.2%

  expected return 13.0% | volatility 20.1% | Sharpe 0.45 | effective positions 5.1

Orders to place
  On public:
    BUY   GLD    $70.00  0.20463  $342.08  $0.00
    BUY   SCHD   $69.24  0.24748  $279.78  $0.00
    BUY   VTI    $60.95   0.5222  $116.72  $0.00
    BUY   VXUS   $57.00  0.29905  $190.62  $0.00
  On kraken_pro:
    BUY   BTC    $22.38  0.042796  $522.85  $0.10
    BUY   ETH    $20.23  0.044812  $451.55  $0.09

  budget $300.00 | invested $299.81 | fees $0.19 (6 bps) | cash left $0.00

Custody (Trezor)
  WAIT  BTC  $22.38  below the $100 sweep threshold; let it grow, then
                     withdraw once in a single larger transfer

Reality check
  On $299.81 invested, the modelled expected return is about $38.88 a year,
  and a typical year swings roughly +/-$60.19.
  You paid $0.19 in fees to open this. That is 0.06% -- earned back only
  after roughly 2 days of the modelled return.
  ...
```

(Run against the `synthetic` provider, so the prices are fake but the machinery
is real.)

## Commands

| Command | What it does |
| --- | --- |
| `ainvest init` | Write a starter `portfolio.yaml` |
| `ainvest optimize` | Target weights plus the orders to reach them |
| `ainvest check` | **The monthly ritual**: contribution plan, tax review, signals, custody — one command |
| `ainvest rebalance` | Trades to steer existing holdings back toward target |
| `ainvest project` | Monte Carlo wealth bands over 1/5/10 years; lump-sum vs DCA |
| `ainvest backtest` | Walk the objective forward through history |
| `ainvest explain` | Returns, volatilities, correlations, objective reference |
| `ainvest fetch` | Refresh or inspect the local price cache |

Useful flags: `--budget`, `--objective`, `--profile`, `--json`, `--markdown`,
`--offline`, `--refresh`, `backtest --exit-rule trend`, `project --dca-months 6`.

`--json` output is stable and meant for scripting.

## Objectives

| Objective | Uses return estimates? | Notes |
| --- | --- | --- |
| `risk_parity` **(default)** | No | Every holding contributes equally to portfolio risk. Depends only on the covariance matrix, so it's much steadier on a short history. |
| `min_vol` | No | The calmest portfolio available. |
| `max_sharpe` | Yes | Best return per unit of risk. Concentrates hard, because it believes the return estimates. |
| `target_vol` | Yes | Most return available at a chosen volatility. |
| `equal_weight` | No | Same weight in everything. A genuinely hard benchmark to beat out of sample. |
| `max_return` | Yes | Included so you can see what over-fitting looks like. |

**Why `risk_parity` is the default.** Expected returns estimated from two years
of daily data are extremely noisy — far noisier than volatilities and
correlations from the same sample. Mean-variance optimization takes those noisy
estimates at face value and responds by concentrating into whichever asset the
noise happened to favour. Risk parity never looks at them.

Three further guards, all configurable:

- **`covariance_shrinkage: ledoit_wolf`** pulls the sample covariance toward a
  well-conditioned target. Without it, a near-singular covariance matrix lets
  the optimizer take huge offsetting positions on the strength of estimation
  noise.
- **`mean_shrinkage`** (default `0.5`) pulls expected returns toward their
  cross-sectional average. At `1.0` the return estimates are ignored entirely.
- **`blend_equal_weight`** (default `0.2`) mixes the optimized answer 20% toward
  equal weight — a little in-sample optimality traded for out-of-sample
  robustness.

## Configuration

`portfolio.yaml` drives everything. The full annotated version is
[`config/portfolio.example.yaml`](config/portfolio.example.yaml).

### Venues and fees

**The shipped fee numbers are realistic estimates, not quoted rates.** Fee
schedules change and vary by account, order type, and volume. Look up what your
venue charges *you* and put the real numbers in. At this account size, this is
the single highest-leverage section of the file.

```yaml
venues:
  public:
    fee_bps: 0
    min_notional: 1
    fractional: true

  coinbase_advanced:
    fee_bps: 60           # advanced/limit orders
    spread_bps: 5
    supports_withdrawal: true
    withdrawal_fee: 1.50

  coinbase_simple:
    spread_bps: 150       # the one-click flow's markup
    supports_withdrawal: true
```

Fees are modelled as `fixed + notional × (fee_bps + spread_bps) / 10000`, floored
at `min_fee`. Use `spread_bps` for venues that mark up the price instead of
charging a visible commission — the cost is real either way.

Each asset lists the venues it can be bought on, and the planner picks the
cheapest one **that will accept an order of that size**. A venue with a $100
minimum is not cheap for a $20 order.

### Constraints

```yaml
constraints:
  single_max: 0.35        # no position above 35%
  crypto_max: 0.25        # total crypto ceiling
  include_threshold: 0.02 # weights under 2% are dropped, not bought as dust
```

Per-asset caps come from both `constraints.single_max` and each asset's own
`max_weight`; the tighter one wins. Keep them loose enough to leave the
optimizer room — if the caps barely add up to 100%, there is only one feasible
portfolio and every objective returns it.

Risk profiles (`conservative`, `balanced`, `aggressive`, `degen`) set
`crypto_max`, `single_max`, and `target_vol` in one go; anything you write in
`constraints:` overrides the profile.

### Rebalancing

The default is **cash-flow rebalancing**: new money is directed at whatever is
furthest below target, and nothing is ever sold.

```yaml
rebalance_band: 0.05   # ignore drift smaller than this
min_trade: 5           # don't place trades below this
allow_sells: false     # true enables two-sided rebalancing
```

Selling to rebalance costs fees on the sell side and, in a taxable account,
realizes gains. For a few hundred dollars that is almost always worse than
simply pointing the next contribution at the underweight position. Turn
`allow_sells` on deliberately, not by default.

### Custody

```yaml
custody:
  wallet: Trezor
  sweep_threshold: 100    # don't withdraw a position smaller than this
  max_fee_fraction: 0.02  # or one whose network fee exceeds 2% of it
```

Per crypto position the tool says `sweep`, `wait`, or `leave`, with the reason.
Assets are only ever recommended for withdrawal if you marked them
`self_custody: true` **and** the venue is configured with
`supports_withdrawal: true` — some retail venues sell price exposure rather than
the coin itself, and a balance you cannot withdraw is not self-custody.

## The steady-investment loop

The intended workflow is one command a month:

```bash
ainvest check
```

which answers, in order: where does this month's contribution go (always toward
whatever is furthest below target — rebalancing with new money instead of
sales), what does your tax position look like, what are the signals saying,
and is anything worth sweeping to the Trezor.

### Taxes: lots, harvesting, and wash sales

Holdings are **tax lots** — one entry per purchase with its cost basis and
date — because that's the level tax reality operates at. The same symbol can
hold a harvestable loss and an untouchable gain at once. With lots recorded,
`check` reports:

- **Harvest candidates**: lots underwater by more than `harvest_min_loss`,
  with the estimated tax saving at your marginal rate, and what to do during
  the 30-day wash window (swap into the configured `tlh_replacement`, e.g.
  VTI→VOO, or sit out).
- **Wash-sale conflicts**: a loss you *can't* harvest yet because a recent buy
  of the same asset sits inside the window — with the number of days until it
  clears. Crypto is treated as wash-exempt (current US treatment) unless you
  set `tax.crypto_wash_sale: true`; the output reminds you that this is a
  recurring legislative target.
- **Long-term boundary warnings**: a gain that would be taxed at 24% today but
  15% in 20 days gets a "wait 20 days, save $X" line before you sell it.

When selling *is* enabled, every proposed sale's estimated tax (HIFO lot
order) is counted as part of the trade's cost, and a sale whose tax swamps the
correction is skipped with the reason printed.

Set both tax rates to zero for a sheltered account and all of this stands down.

### The fee-payback gate

Every rebalancing trade must earn its fee back within `fee_payback_months`
(default 6) at the portfolio's expected return, or it is skipped with the
arithmetic shown. This is the formal version of "don't make trades that cost
more in fees than they can plausibly return."

### "When to put money in"

Steadily, per the configured schedule — that is the whole engine. Two optional
refinements, both visible in `check`:

- `contribution.reserve_fraction` holds back a slice of each contribution as
  dry powder, flagged for deployment when the portfolio falls
  `deploy_on_drawdown` below its peak (buying more when things are down,
  systematically rather than emotionally).
- `ainvest project --dca-months 6` compares deploying a lump sum now versus
  spreading it, on your own portfolio's bootstrapped history. History usually
  favours *sooner*; the output says by how much and how often.

### "When to pull out" — the honest answer

The tool computes a trend signal (portfolio vs its 200-day average) and a
drawdown alert, and it will **never** turn either into a sell order. Instead:

```bash
ainvest backtest --exit-rule trend
```

runs the exit rule against staying invested **with taxes and fees charged on
every exit** — exits realise gains at up to your short-term rate, while
staying in defers a single long-term bill — and liquidates both at the end so
unrealised gains can't flatter either side. On most histories the exit rule
loses after costs. If it loses on yours, you have your answer; if it wins, you
at least know what the bet is. The drawdown alert exists for the opposite
reason: historically, well below the peak is where contributions buy the most.

### News and events — context, never triggers

`check` includes a **News & events** section with three layers:

- **Market stress gauge** (offline, quantitative): the portfolio's realised
  volatility versus its own history. When it's in the top decile, the output
  says so — and says the plan doesn't change, because news that matters shows
  up in prices, and selling into stress is how drawdowns become losses.
- **Your calendar** (`news.events` in the config): FOMC dates, CPI prints, an
  expected protocol upgrade — listed with days-to-go so volatility around
  them is unsurprising rather than actionable.
- **Headlines** (network, best-effort): per-asset RSS via Yahoo Finance. A
  dead network degrades to "no headlines", never to a failed check; `--offline`
  skips the fetch entirely.

The design stance is explicit in the code and the output: no headline changes
a single order. Reacting to news is one of the best-documented ways retail
investors underperform their own holdings, so this tool shows you the context
and then tells you it's context.

### The withdrawal policy — money never leaves without consent

A hard rule, not a preference: plans will **never propose selling more than
`withdrawal.auto_limit` dollars per run** (default **$0**) unless you pass
`--approve-sells` for that specific run. Deposits are the mirror image — the
programmed `contribution` schedule is the only inflow the tool plans around.

Sells beyond the allowance are *withheld*, listed with a "pending your
approval" reason, and the buy side shrinks to match — buys can only spend new
cash plus what the permitted sells actually raised. Approving is always a
deliberate act:

```bash
ainvest rebalance --allow-sells                  # sells capped at the allowance
ainvest rebalance --allow-sells --approve-sells  # you said so, this run only
```

### Long-run expectations

```bash
ainvest project --years 10
```

block-bootstraps your portfolio's own returns into thousands of paths with
your contribution schedule applied, and prints wealth bands (p10 through p90),
the probability of ending below what you paid in, and the median worst
drawdown along the way. The p10 column is printed at the same width as the
p90 column on purpose.

## Data

| Provider | Use |
| --- | --- |
| `yahoo` | Stocks and crypto from one source (needs `pip install '.[live]'`) |
| `csv` | Your own `<SYMBOL>.csv` files with `date,close` columns |
| `synthetic` | A deterministic fake market for offline use and tests |

Prices are cached as plain CSV under `data.cache_dir` — greppable, inspectable,
and hand-editable. If a fetch fails, the tool falls back to cached data and says
so rather than refusing to run.

**Calendars.** Crypto trades every day; equities don't. Prices are intersected
onto the dates where every asset has one, so each row spans the same interval
for all of them and a weekend crypto move isn't misattributed. The
annualization factor is then derived from the observed sampling density, which
comes out near 252 for a mixed portfolio and 365 for a crypto-only one.

## Some things worth knowing

**The tool refuses to make several mistakes for you**, and each refusal is
explained in the output:

- It won't place an order below a venue's minimum; it redistributes that money.
- It won't spend more than the budget — order sizes round *down*, and fees come
  out of the budget rather than on top of it.
- It won't recommend a hardware-wallet sweep whose network fee eats more than
  `max_fee_fraction` of the position.
- It won't run a backtest without enough history to hold half of it out of
  sample.
- It tells you when constraints are impossible, and which knob to turn.
- It won't propose pulling out more than the programmed withdrawal allowance
  without an explicit per-run approval flag.
- It won't turn a headline, a calendar entry, or its own stress gauge into a
  trade.

**And the honest limitations:**

- Historical estimates are the only input. Two years of data is a small sample,
  and the future is not obliged to resemble it.
- Tax analysis is a two-rate estimate on the lots you record. It ignores
  netting order, thresholds, state tax, and every purchase this config can't
  see (other accounts, spousal accounts, IRAs — all of which count for wash
  sales). It flags conversations to have, it does not replace having them.
- The backtest is a sanity check on the *method*, not a forecast. It has no
  slippage model and assumes you can transact at closing prices.
- At a few hundred dollars, how much you contribute and whether you leave it
  alone will dominate anything in here. The tool says so in its own output.

## Development

```bash
pip install -e '.[dev]'
pytest                 # 232 tests, no network needed
ruff check .
```

Tests run entirely on the synthetic provider, so they're deterministic and
offline. Notable ones:

- `test_projection.py` verifies the constrained projection against its
  variational-inequality characterization — the property that defines *the*
  projection rather than merely a feasible point.
- `test_objectives.py` checks each objective against a 30,000-sample brute-force
  search, and verifies every analytic gradient against finite differences.
- `test_backtest.py::test_no_lookahead` rewrites all prices after a cut-off date
  and asserts the equity curve before it doesn't move by a basis point; the
  trend overlay has its own equivalent, plus a proof that in a straight-up
  market the exit rule can only tie or lose after costs.
- `test_lots.py` pins the wash-sale window arithmetic, HIFO ordering, the
  long-term boundary maths, and the crypto exemption toggle.
- `test_rebalance.py::TestWithdrawalPolicy` proves sells are withheld at the
  default $0 allowance, capped at a programmed one, and released only by
  explicit approval; `TestFundedBuys` proves buys can never outspend new cash
  plus actual sell proceeds.

## License

MIT — see [LICENSE](LICENSE).

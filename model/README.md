# Mathematical reconstruction core

These modules are app-local copies of the project's chat mathematical model v3.
The original notebook project is left unchanged. Source SHA-256 values are in
`provenance.json`.

The model is a marked compound-Poisson arrival process with fixed lognormal (or
lognormal-mixture) willingness-to-pay assumptions, Gamma regularisation of
arrival intensities, and exact conditioning on an external primary-demand seat
total. It is a mathematical likelihood model, not a trained ML predictor.

`prepare_inputs_v3` validates operation logs and complete supplied offer intervals.
With purchases but no offer log, purchase-derived prices and unlimited
availability remain explicit assumptions. With no purchases, complete offers,
an external all-demand group-size distribution and an explicit log-budget prior
are required. No observed bookings are manufactured. No-sale strata and flights
are labelled as dependent on external priors and the supplied total.

`fit_model_v3` keeps raw fitted hidden intensities in `marks`; use these with
`conditional_metrics_v3.conditional_seat_pmf` and `sample_passengers_v3`. Feeding
posterior group means back into those functions would condition twice.

The app copy corrects two reporting fields: `median_amount` is the actual mixture
median at DtD=30, group size 1 and zero cabin offset; `budget_scale=exp(mu_log)` is
the base component scale. `unconditioned_expected_primary_seats` is expected
purchased seats plus raw expected hidden seats. The former hybrid, observed gross
sales plus raw hidden expectation, is preserved under
`observed_gross_plus_unconditioned_hidden`.

Generated hidden passengers, budgets, rejection reasons and preferred cabin
labels are model samples, not recovered individuals. Exact count conditioning
does not identify them. Conditional 80% intervals hold fitted parameters fixed
and do not include uncertainty about the behavioural assumptions or arrival
parameters. Passenger reference files are never read by these modules.

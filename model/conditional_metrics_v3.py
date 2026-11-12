"""Exact marginal seat uncertainty conditional on a known compound-Poisson sum.

These are conditional distributions with fitted MAP parameters held fixed.
They are NOT posterior intervals integrating behavioral/arrival parameters.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp


def _rate_array(rates,max_q=None):
    if not rates:
        return np.zeros(max_q or 1)
    qmax=max(max(map(int,rates)),max_q or 1)
    answer=np.zeros(qmax)
    for q,value in rates.items():
        if int(q)!=q or q<1 or not np.isfinite(value) or value<0:
            raise ValueError('Finite nonnegative intensities at positive integer sizes required.')
        answer[int(q)-1]=float(value)
    return answer


def _tilted_pmf(rates,total):
    """CP probabilities 0..total for already globally tilted rates."""
    quantities=np.arange(1,len(rates)+1)
    if total<600:
        values=np.zeros(total+1);values[0]=np.exp(-rates.sum())
        for n in range(1,total+1):
            take=quantities<=n
            values[n]=np.dot(quantities[take]*rates[take],values[n-quantities[take]])/n
        return values
    logs=np.full(total+1,-np.inf);logs[0]=-rates.sum()
    positive=rates>0
    for n in range(1,total+1):
        take=(quantities<=n)&positive
        if take.any():
            logs[n]=logsumexp(np.log(quantities[take]*rates[take])+logs[n-quantities[take]])-np.log(n)
    return logs


def conditional_seat_pmf(region_rates,other_rates,total):
    """P(region seats=k | region seats+other seats=total), k=0..total.

    Shared exponential tilt cancels in the conditional distribution. This also
    handles a completely closed region, impossible totals, and empty regions.
    """
    if int(total)!=total or total<0:
        raise ValueError('The conditioned seat total must be a nonnegative integer.')
    total=int(total)
    if total==0:
        return np.array([1.])
    n=max(max(map(int,region_rates),default=1),max(map(int,other_rates),default=1))
    a=_rate_array(region_rates,n);b=_rate_array(other_rates,n)
    combined=a+b;positive=combined>0
    if not positive.any():
        raise ValueError('Positive total has no compound-Poisson support.')
    quantities=np.arange(1,len(combined)+1)
    def equation(t):
        return logsumexp(np.log(quantities[positive]*combined[positive])+quantities[positive]*t)-np.log(total)
    lower,upper=-1.,1.
    for _ in range(12):
        if equation(lower)<=0<=equation(upper): break
        lower*=2;upper*=2
    tilt=brentq(equation,lower,upper)
    factor=np.exp(quantities*tilt)
    ar=_tilted_pmf(a*factor,total);br=_tilted_pmf(b*factor,total)
    if total<600:
        values=ar*br[::-1]
        if not np.isfinite(values).all() or values.sum()<=0:
            raise ValueError('The requested total is unreachable with supported group sizes.')
        return values/values.sum()
    logs=ar+br[::-1]
    normalization=logsumexp(logs)
    if not np.isfinite(normalization):
        raise ValueError('The requested total is unreachable with supported group sizes.')
    return np.exp(logs-normalization)


def discrete_scores(pmf,actual):
    if int(actual)!=actual or actual<0 or actual>len(pmf)-1:
        raise ValueError('Truth is outside the conditioned support.')
    values=np.asarray(pmf,float)
    cdf=np.cumsum(values)
    actual=int(actual)
    low=int(np.searchsorted(cdf,.10));high=int(np.searchsorted(cdf,.90))
    grid=np.arange(len(values))
    return {'conditional_mean':float(values@grid),'p10':low,'p90':high,
            'actual':actual,'CRPS':float(np.square(cdf-(grid>=actual)).sum()),
            'covered80':bool(low<=actual<=high),'interval_width':high-low}


def conditional_period_metrics(run,result):
    marks=result['marks']
    # marks must contain UNCONDITIONED fitted hidden intensities, not conditional
    # group means; replacing them would condition twice and change the model.
    hidden=marks.groupby(['period','qty']).expected_hidden_groups.sum()
    total_rates=marks.groupby('qty').expected_hidden_groups.sum().to_dict()
    gross=int(run.bookings.commercial.clip(lower=0).sum())
    target=int(run.primary_demand_seats)-gross
    truth=run.passengers.loc[run.passengers.outcome.eq('waste')].groupby('period').qty.sum()
    periods=list(dict.fromkeys(result['strata'].period.astype(str)))
    rows=[]
    for period in periods:
        region=hidden.loc[period].to_dict() if period in hidden.index.get_level_values(0) else {}
        other={q:max(0.,value-region.get(q,0.)) for q,value in total_rates.items()}
        pmf=conditional_seat_pmf(region,other,target)
        rows.append({'period':period,**discrete_scores(pmf,int(truth.get(period,0)))})
    return pd.DataFrame(rows)


def score_uncertainty(run,result):
    table=conditional_period_metrics(run,result)
    return {'conditional_CRPS_mean':float(table.CRPS.mean()),
            'conditional_coverage80':float(table.covered80.mean()),
            'conditional_width80_mean':float(table.interval_width.mean()),
            'uncertainty_scope':'latent_CP_given_total_and_fixed_MAP_parameters'}

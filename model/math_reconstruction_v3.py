"""Joint marked-Poisson reconstruction with exact latent-seat conditioning.

No passenger reference file is read. Behaviour and pooled arrival priors must be
supplied independently. This version never forces expected demand to equal the
realised external seat total. It uses the likelihood of that realised count.

``observed_total`` means N is a random realised primary-seat count observed in
addition to bookings. ``fixed_design`` divides by P(primary seats=N), as required
when the compound-Poisson generator is conditioned on N before observations.
The distinction is substantive. Cancellation identities and individual latent
budgets are not recovered facts. Offered states are taken as given; the model
requires that unrecorded refusals do not change the observed policy/history.
"""
from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri_exp

from .math_reconstruction_v2 import (
    ModelConfig, STRATUM_KEYS, REASONS, _model_arrays, _allocate_integer,
    _conditional_group_counts, _draw_amount, prepare_inputs_v2,
)


def prepare_inputs_v3(bookings, departure_date, horizon_start, horizon_end,
                      offers=None, path_method="last"):
    """Prepare observed operations without manufacturing purchases.

    With complete supplied offers an empty/cancel-only journal is valid. Fitting
    it additionally requires external demand, quantity and budget assumptions;
    those assumptions do not become identified passenger characteristics.
    """
    return prepare_inputs_v2(bookings, departure_date, horizon_start, horizon_end,
                             offers=offers, path_method=path_method,
                             allow_empty_purchases=offers is not None)


def _mixture_median(mu, config):
    """Median in currency units; exp(mu) is only the base component scale."""
    weights, offsets, sigmas = _mixture(config)
    if len(weights) == 1:
        location = mu + float(offsets[0])
    else:
        center = brentq(lambda value: float(np.dot(weights, ndtr((value-offsets)/sigmas)))-.5,
                        float(np.min(offsets-12*sigmas)), float(np.max(offsets+12*sigmas)))
        location = mu + center
    return float(np.exp(location)) if location < 709 else np.inf


@dataclass
class ArrivalPrior:
    """Gamma prior with mode mean_rates, rate strength_days.

    Keys: (flight,cabin,period), (cabin,period), cabin, or 'global'. Values are
    primary GROUPS/day. Shape is 1+strength_days*mean_rate, so the prior *mode*
    equals mean_rate. With strength_days=0 there is no arrival regularisation;
    fully closed/no-sale cells then remain unidentified despite numeric bounds.
    A missing mean uses a neutral external-N based initial rate, explicitly
    labelled in the result rather than disguised as independently learned data.
    """
    mean_rates: object = None
    strength_days: float = 2.0


@dataclass
class BehaviorConfigV3(ModelConfig):
    """Optional residual log-WTP mixture; offsets are relative to run mu.

    Empty mixture_sigmas uses config.sigma for every component. Setting a
    single weight 1 and offset 0 exactly reproduces the lognormal family.
    """
    mixture_weights: tuple = (1.0,)
    mixture_offsets: tuple = (0.0,)
    mixture_sigmas: tuple = ()


def _mixture(config):
    w=np.asarray(getattr(config,'mixture_weights',(1.0,)),float)
    o=np.asarray(getattr(config,'mixture_offsets',(0.0,)),float)
    sig=getattr(config,'mixture_sigmas',())
    s=np.asarray(sig if len(sig) else [config.sigma]*len(w),float)
    if (len(w)==0 or len(o)!=len(w) or len(s)!=len(w)
            or not np.isfinite(w).all() or not np.isfinite(o).all() or not np.isfinite(s).all()
            or (w<=0).any() or (s<=0).any()):
        raise ValueError('Mixture weights/sigmas must be positive and all mixture vectors finite and equal length.')
    return w/w.sum(),o,s


def compound_poisson_stats(intensities, total_seats):
    """Exact log P(sum q*Kq=n) and E[Kq|sum q*Kq=n].

    Kq are independent Poisson(nu_q). Exponential tilting moves the recurrence
    near its mode, avoiding huge unnormalised coefficients. Derivative with
    respect to nu_q is E[Kq|n]/nu_q - 1. Positive q above n still contribute
    -nu_q to the log probability. Unsupported totals return logpmf=-inf.
    """
    if int(total_seats)!=total_seats or total_seats<0:
        raise ValueError('Conditional seat total must be a nonnegative integer.')
    n=int(total_seats)
    original={int(q):float(v) for q,v in intensities.items()}
    if any(int(q)!=q or q<=0 for q in intensities) or any(not np.isfinite(v) or v<0 for v in original.values()):
        raise ValueError('Compound-Poisson marks require positive integer quantities and finite nonnegative intensities.')
    qs=np.array(sorted(original),dtype=int)
    nu=np.array([original[int(q)] for q in qs],float)
    expected=np.zeros(len(qs),float)
    zero_ratios=np.zeros(len(qs),float)
    if n==0:
        return {'logpmf':-float(nu.sum()),'expected_groups':dict(zip(qs.tolist(),expected.tolist())),
                'gradient_intensity':{int(q):-1.0 for q in qs},'tilt':0.0}
    positive=nu>0
    if not positive.any():
        return {'logpmf':-np.inf,'expected_groups':dict(zip(qs.tolist(),expected.tolist())),
                'gradient_intensity':{int(q):np.nan for q in qs},'tilt':np.nan}
    qpos=qs[positive];npos=nu[positive]
    def mean_at(t):
        return float(np.dot(qpos,npos*np.exp(np.clip(qpos*t,-745,700))))-n
    lo,hi=-80.0,80.0
    while mean_at(lo)>0: lo*=2
    while mean_at(hi)<0: hi*=2
    tilt=brentq(mean_at,lo,hi,xtol=1e-12)
    tilted=np.zeros_like(nu)
    tilted[positive]=npos*np.exp(qpos*tilt)
    use=(qs<=n)&positive
    q=qs[use];rates=tilted[use]
    if len(q)==0:
        return {'logpmf':-np.inf,'expected_groups':dict(zip(qs.tolist(),expected.tolist())),
                'gradient_intensity':{int(v):np.nan for v in qs},'tilt':tilt}
    total_tilt=float(tilted.sum())
    if total_tilt<650:
        p=np.zeros(n+1,float);p[0]=np.exp(-total_tilt)
        for k in range(1,n+1):
            active=q<=k
            p[k]=np.dot(q[active]*rates[active],p[k-q[active]])/k
        if p[n]<=0:
            return {'logpmf':-np.inf,'expected_groups':dict(zip(qs.tolist(),expected.tolist())),
                    'gradient_intensity':{int(v):np.nan for v in qs},'tilt':tilt}
        log_last=float(np.log(p[n]))
        for j,mark in enumerate(qs):
            if mark<=n and nu[j]>0:
                expected[j]=tilted[j]*p[n-mark]/p[n]
            elif mark<=n:
                zero_ratios[j]=np.exp(np.clip(mark*tilt,-745,709))*p[n-mark]/p[n]
    else:
        # Large user totals use log recurrence. The usual research fixtures are
        # N~300 and stay in the faster linear, tilted-probability branch.
        lp=np.full(n+1,-np.inf);lp[0]=-total_tilt
        for k in range(1,n+1):
            active=q<=k
            lp[k]=logsumexp(np.log(q[active]*rates[active])+lp[k-q[active]])-log(k)
        log_last=float(lp[n])
        if not np.isfinite(log_last):
            return {'logpmf':-np.inf,'expected_groups':dict(zip(qs.tolist(),expected.tolist())),
                    'gradient_intensity':{int(v):np.nan for v in qs},'tilt':tilt}
        for j,mark in enumerate(qs):
            if mark<=n and nu[j]>0:
                expected[j]=tilted[j]*np.exp(lp[n-mark]-log_last)
            elif mark<=n:
                zero_ratios[j]=np.exp(np.clip(mark*tilt+lp[n-mark]-log_last,-745,709))
    logpmf=log_last+float(tilted.sum()-nu.sum())-n*tilt
    grad=np.divide(expected,nu,out=np.zeros_like(expected),where=nu>0)-1
    grad[~positive]=zero_ratios[~positive]-1
    return {'logpmf':float(logpmf),'expected_groups':dict(zip(qs.tolist(),expected.tolist())),
            'gradient_intensity':dict(zip(qs.tolist(),grad.tolist())),'tilt':float(tilt)}


def _survival(mu,threshold,config):
    weights,offsets,sigmas=_mixture(config)
    z=(mu+offsets[:,None]-threshold[None,:])/sigmas[:,None]
    sf=np.sum(weights[:,None]*ndtr(z),axis=0)
    below=np.sum(weights[:,None]*ndtr(-z),axis=0)
    derivative=np.sum(weights[:,None]*np.exp(-.5*z*z)/(sqrt(2*np.pi)*sigmas[:,None]),axis=0)
    return sf,below,derivative


def _event_logbuy(mu,threshold,config):
    w,o,s=_mixture(config)
    z=(mu+o[:,None]-threshold[None,:])/s[:,None]
    logs=logsumexp(np.log(w)[:,None]+log_ndtr(z),axis=0)
    log_derivative=logsumexp(np.log(w)[:,None]-.5*z*z-np.log(s)[:,None]-.5*log(2*np.pi),axis=0)
    return logs+log(1-config.other_reject_probability),np.exp(log_derivative-logs)


def _rate_means(rows,inputs,prior,primary_target,group_distributions):
    if prior.strength_days<0 or not np.isfinite(prior.strength_days):
        raise ValueError('Arrival prior strength_days must be finite and nonnegative.')
    # Neutral initial exposure is one process per cabin, so sum exposure over
    # cabins. This is an external-volume initial guess, not hidden truth.
    mean_qty=float((group_distributions.qty*group_distributions.probability).sum()/len(rows))
    neutral=max(float(primary_target)/(max(mean_qty,1)*rows.exposure_days.sum()),1e-6)
    rates=[];sources=[]
    mapping=prior.mean_rates
    if mapping is not None and not isinstance(mapping,Mapping):
        raise ValueError('Arrival prior mean_rates must be a mapping or None.')
    for row in rows.itertuples():
        value=None
        if mapping is not None:
            for key in ((row.flight,row.cabin,row.period),(row.cabin,row.period),row.cabin,'global'):
                if key in mapping: value=float(mapping[key]);break
        rates.append(neutral if value is None else value)
        sources.append('neutral_external_N_initial_scale' if value is None else 'supplied_pooled_prior')
    answer=np.asarray(rates,float)
    if not np.isfinite(answer).all() or (answer<=0).any():
        raise ValueError('Every arrival prior mode must be finite and strictly positive.')
    return answer,sources


def _flight_problem(inputs,full,flight,config,target,prior,mu_prior,total_mode):
    rows=inputs.strata.loc[inputs.strata.flight.eq(flight)].copy().reset_index(drop=True)
    remap={old:i for i,old in enumerate(rows.stratum_id)}
    take=full['marks'].flight.eq(flight)
    base=full['marks'].loc[take].copy().reset_index(drop=True)
    idx=np.array([remap[int(v)] for v in base.stratum_id],int)
    qs=np.asarray(config.qty_values,int)
    qmap={q:i for i,q in enumerate(qs)}
    qi=np.array([qmap[int(q)] for q in base.qty],int)
    lengths=base.duration_days.to_numpy(float)*base.probability.to_numpy(float)
    available=base.open.to_numpy(bool)&(base.qty.to_numpy(int)<=base.available_seats.to_numpy(float))
    safe_price=np.where(base.open.to_numpy(bool),base.price.to_numpy(float),1.)
    offset=(config.beta_time*np.log((1+base.dtd.to_numpy(float))/31)
            +base.cabin.map(config.cabin_offsets or {}).fillna(0).to_numpy(float)
            +config.beta_qty*np.log(base.qty.to_numpy(float)))
    threshold=np.log(safe_price)-offset
    mark_index=idx*len(qs)+qi
    total_exposure=np.bincount(mark_index,weights=lengths,minlength=len(rows)*len(qs)).reshape(len(rows),len(qs))
    gd=full['group_distributions'].loc[full['group_distributions'].flight.eq(flight)]
    mean_rates,prior_sources=_rate_means(rows,inputs,prior,target,gd)
    buys=inputs.purchases.loc[inputs.purchases.flight.eq(flight)].copy()
    obs=buys.merge(gd[STRATUM_KEYS+['qty','probability','stratum_id']],on=STRATUM_KEYS+['qty'],how='left',validate='many_to_one')
    event_idx=np.array([remap[int(v)] for v in obs.stratum_id],int)
    event_threshold=(np.log(obs.price.to_numpy(float))-config.beta_time*np.log((1+obs.dtd.to_numpy(float))/31)
        -obs.cabin.map(config.cabin_offsets or {}).fillna(0).to_numpy(float)-config.beta_qty*np.log(obs.qty.to_numpy(float)))
    counts=rows.purchase_events.to_numpy(float)
    gross=int(buys.qty.sum());hidden=int(target)-gross
    if hidden<0: raise ValueError('Primary demand is smaller than observed gross sold seats.')
    muprior=mu_prior.get(flight) if isinstance(mu_prior,Mapping) else mu_prior
    if muprior is None and buys.empty:
        raise ValueError(f"No purchases for {flight}: supply an explicit independent mu_prior=(log_budget, log_sd).")
    if muprior is None:
        muprior=(float(np.log(buys.price.astype(float).median())),2.0)
        muprior_source='broad_data_scaled_assumption_not_independent_training'
    else: muprior_source='supplied_independent_behavior_prior'
    mu_center,mu_sd=map(float,muprior)
    if not np.isfinite(mu_center) or not np.isfinite(mu_sd) or mu_sd<=0:
        raise ValueError('mu_prior must be (finite log-WTP location, positive standard deviation).')
    strength=float(prior.strength_days)
    r=config.other_reject_probability

    def evaluate(x,details=False):
        rates=np.exp(x[:-1]);mu=float(x[-1])
        survival,below,ds=_survival(mu,threshold,config)
        pbuy=available*survival*(1-r)
        phidden=(~available).astype(float)+available*(below+survival*r)
        dp=available*ds*(1-r)
        effective=np.bincount(idx,weights=lengths*pbuy,minlength=len(rows))
        deffective=np.bincount(idx,weights=lengths*dp,minlength=len(rows))
        h_exposure=np.bincount(mark_index,weights=lengths*phidden,minlength=len(rows)*len(qs)).reshape(len(rows),len(qs))
        dh_exposure=-np.bincount(mark_index,weights=lengths*dp,minlength=len(rows)*len(qs)).reshape(len(rows),len(qs))
        nu=(rates[:,None]*h_exposure).sum(axis=0)
        hidden_stats=compound_poisson_stats(dict(zip(qs,nu)),hidden)
        if not np.isfinite(hidden_stats['logpmf']):
            return (np.inf,np.zeros_like(x)) if not details else None
        logbuy,event_dlog=_event_logbuy(mu,event_threshold,config)
        obsll=float(np.dot(counts,x[:-1])+np.log(obs.probability.to_numpy(float)).sum()+logbuy.sum()-np.dot(rates,effective))
        ll=obsll+hidden_stats['logpmf']
        hidden_grad=np.array([hidden_stats['gradient_intensity'][int(q)] for q in qs])
        grad_z=counts-rates*effective+rates*(h_exposure@hidden_grad)
        grad_mu=float(event_dlog.sum()-np.dot(rates,deffective)+np.dot((rates[:,None]*dh_exposure).sum(axis=0),hidden_grad))
        total_stats=None
        if total_mode=='fixed_design':
            total_nu=(rates[:,None]*total_exposure).sum(axis=0)
            total_stats=compound_poisson_stats(dict(zip(qs,total_nu)),target)
            if not np.isfinite(total_stats['logpmf']):
                return (np.inf,np.zeros_like(x)) if not details else None
            ll-=total_stats['logpmf']
            tgrad=np.array([total_stats['gradient_intensity'][int(q)] for q in qs])
            grad_z-=rates*(total_exposure@tgrad)
        prior_ll=float(np.dot(strength*mean_rates,x[:-1])-strength*rates.sum()-.5*((mu-mu_center)/mu_sd)**2)
        grad_z+=strength*(mean_rates-rates)
        grad_mu-=(mu-mu_center)/mu_sd**2
        if not details: return -(ll+prior_ll),-np.r_[grad_z,grad_mu]
        return {'rates':rates,'mu':mu,'survival':survival,'below':below,'derivative':ds,
                'buy':pbuy,'hidden_probability':phidden,'effective':effective,
                'h_exposure':h_exposure,'nu_hidden':nu,'hidden_stats':hidden_stats,
                'total_stats':total_stats,'log_likelihood':ll,'observation_log_likelihood':obsll,
                'log_prior':prior_ll,'log_posterior':ll+prior_ll}
    _,_,sigmas=_mixture(config)
    margin=max(6.,8*float(sigmas.max()))
    boundary_values=np.r_[threshold,event_threshold,mu_center]
    low_mu=float(boundary_values.min())-margin
    high_mu=float(boundary_values.max())+margin
    lower_rate=np.log(1e-9);upper_rate=np.log(max(target,10)*10.)
    bounds=[(lower_rate,upper_rate)]*len(rows)+[(low_mu,high_mu)]
    initial=np.r_[np.log(mean_rates),mu_center]
    return {'rows':rows,'base':base,'idx':idx,'qi':qi,'quantities':qs,'lengths':lengths,
            'offset':offset,'available':available,'total_exposure':total_exposure,
            'mean_rates':mean_rates,'prior_sources':prior_sources,'mu_prior_source':muprior_source,
            'mu_prior':(mu_center,mu_sd),'hidden_target':hidden,'evaluate':evaluate,
            'initial':initial,'bounds':bounds,'counts':counts}


def fit_model_v3(inputs,primary_demand_seats=300,config=None,prior=None,
                 total_mode='observed_total',mu_prior=None,maxiter=150):
    """Joint MAP of rates/run-WTP intercept using observed events and exact N.

    MAP is defined w.r.t. rates lambda (not log lambda); optimisation uses a
    log transform but deliberately adds no Jacobian. Behaviour coefficients,
    qty distribution and mixture parameters remain externally fixed. With no
    arrival prior some components may be unidentified; bounds are reported.
    Mixture/profile selection must happen on independent training/validation.
    """
    if total_mode not in ('observed_total','fixed_design'):
        raise ValueError("total_mode must be 'observed_total' or 'fixed_design'.")
    config=ModelConfig() if config is None else config
    prior=ArrivalPrior() if prior is None else prior
    _mixture(config)
    full=_model_arrays(inputs,config)
    flights=inputs.strata.flight.unique().tolist()
    targets=dict(primary_demand_seats) if isinstance(primary_demand_seats,Mapping) else {f:primary_demand_seats for f in flights}
    output_rows=[];base_parts=[];marks_parts=[];post_parts=[];parameter_rows=[]
    for flight in flights:
        if flight not in targets or not np.isfinite(targets[flight]) or int(targets[flight])!=targets[flight] or targets[flight]<0:
            raise ValueError('Provide a nonnegative integer primary-seat total for every flight.')
        target=int(targets[flight])
        problem=_flight_problem(inputs,full,flight,config,target,prior,mu_prior,total_mode)
        initial=problem['initial']
        # First try the independently supplied / neutral starting point.
        fits=[minimize(problem['evaluate'],initial,jac=True,method='L-BFGS-B',bounds=problem['bounds'],
                       options={'maxiter':int(maxiter),'ftol':1e-10,'gtol':2e-5,'maxls':35})]
        if not fits[0].success or not np.isfinite(fits[0].fun):
            alternative=initial.copy();alternative[-1]-=.5
            fits.append(minimize(problem['evaluate'],alternative,jac=True,method='L-BFGS-B',bounds=problem['bounds'],
                       options={'maxiter':int(maxiter)*2,'ftol':1e-10,'gtol':2e-5,'maxls':45}))
        finite=[fit for fit in fits if np.isfinite(fit.fun)]
        if not finite: raise ValueError('No finite likelihood: exact hidden total is unreachable under supported quantities/behaviour.')
        converged=[fit for fit in finite if fit.success]
        optimum=min(converged if converged else finite,key=lambda fit:fit.fun)
        state=problem['evaluate'](optimum.x,details=True)
        if state is None: raise ValueError('The optimum has no finite conditional latent-count probability.')
        rows=problem['rows'].copy();base=problem['base'].copy();idx=problem['idx'];qi=problem['qi']
        rates=state['rates'];qs=problem['quantities'];lengths=problem['lengths'];mu=state['mu']
        potential=rates[idx]*lengths
        sf,below=state['survival'],state['below']
        a=problem['available'].astype(float);r=config.other_reject_probability
        probs={'availability':1-a,'budget':a*below,'other':a*sf*r}
        base['mu_log']=mu;base['wtp_log_location']=mu+problem['offset']
        base['p_survival']=sf;base['p_capacity']=probs['availability']
        base['p_budget']=probs['budget'];base['p_other']=probs['other'];base['p_buy']=state['buy']
        base['expected_primary_groups']=potential
        base['expected_hidden_groups']=potential*state['hidden_probability']
        base['raw_expected_hidden_seats']=base.expected_hidden_groups*base.qty
        base_parts.append(base)
        conditional_q=np.array([state['hidden_stats']['expected_groups'][int(q)] for q in qs])
        nu=state['nu_hidden']
        correction=np.divide(conditional_q,nu,out=np.zeros_like(nu),where=nu>0)
        posterior_groups=np.zeros(len(rows));posterior_seats=np.zeros(len(rows))
        for reason,probability in probs.items():
            marked=base.drop(columns='expected_primary_groups').copy()
            marked['reason']=reason;marked['reason_probability']=probability
            marked['expected_hidden_groups']=potential*probability
            marked['raw_expected_hidden_seats']=marked.expected_hidden_groups*marked.qty
            marks_parts.append(marked)
            post=marked.copy()
            post['unconditioned_hidden_groups']=marked.expected_hidden_groups
            post['posterior_expected_hidden_groups']=marked.expected_hidden_groups.to_numpy()*correction[qi]
            post['posterior_expected_hidden_seats']=post.posterior_expected_hidden_groups*post.qty
            post_parts.append(post)
            groups=np.bincount(idx,weights=post.posterior_expected_hidden_groups,minlength=len(rows))
            seats=np.bincount(idx,weights=post.posterior_expected_hidden_seats,minlength=len(rows))
            posterior_groups+=groups;posterior_seats+=seats
            rows[f'hidden_seats_{reason}']=seats
        raw_groups=np.bincount(idx,weights=base.expected_hidden_groups,minlength=len(rows))
        raw_seats=np.bincount(idx,weights=base.raw_expected_hidden_seats,minlength=len(rows))
        rows['lambda_groups_per_day']=rates
        rows['prior_mode_groups_per_day']=problem['mean_rates']
        rows['prior_source']=problem['prior_sources']
        rows['effective_exposure_days']=state['effective']
        rows['expected_hidden_groups']=raw_groups
        rows['raw_expected_hidden_seats']=raw_seats
        rows['observed_gross_plus_unconditioned_hidden']=rows.purchased_seats+raw_seats
        rows['posterior_expected_hidden_groups']=posterior_groups
        rows['posterior_expected_hidden_seats']=posterior_seats
        rows['posterior_expected_primary_seats']=rows.purchased_seats+posterior_seats
        rows['estimated_waste_seats']=_allocate_integer(problem['hidden_target'],posterior_seats)
        rows['expected_purchased_groups']=rates*state['effective']
        rows['expected_purchased_seats']=np.bincount(idx,weights=potential*state['buy']*base.qty.to_numpy(float),minlength=len(rows))
        rows['unconditioned_expected_primary_seats']=rows.expected_purchased_seats+raw_seats
        rows['identification']=np.where(rows.purchase_events.gt(0),'model_conditional',
            'prior_and_total_informed_no_sales' if prior.strength_days>0 else 'no_sales_unidentified_without_arrival_prior')
        if abs(posterior_seats.sum()-problem['hidden_target'])>1e-6:
            raise AssertionError('Exact CP conditional expected seats do not sum to the hidden total.')
        output_rows.append(rows)
        bounds=problem['bounds'];bound_hits=sum(abs(v-lo)<1e-5 or abs(v-hi)<1e-5 for v,(lo,hi) in zip(optimum.x,bounds))
        parameter_rows.append({'flight':flight,'mu_log':mu,'budget_scale':float(np.exp(mu)) if mu<709 else np.inf,
            'median_amount':_mixture_median(mu,config),
            'sigma':float(config.sigma),'beta_time':config.beta_time,'beta_qty':config.beta_qty,
            'other_reject_probability':config.other_reject_probability,'primary_demand_seats':target,
            'expected_primary_seats':float(rows.posterior_expected_primary_seats.sum()),
            'unconditioned_expected_primary_seats':float(rows.unconditioned_expected_primary_seats.sum()),
            'observed_gross_plus_unconditioned_hidden':float(rows.observed_gross_plus_unconditioned_hidden.sum()),
            'posterior_hidden_seats':float(posterior_seats.sum()),'hidden_seats_log_probability':state['hidden_stats']['logpmf'],
            'log_likelihood':state['log_likelihood'],'log_prior':state['log_prior'],'log_posterior':state['log_posterior'],
            'optimizer_success':bool(optimum.success),'optimizer_message':str(optimum.message),
            'iterations':int(optimum.nit),'gradient_max_abs':float(np.max(np.abs(optimum.jac))),
            'parameter_bound_hits':int(bound_hits),'zero_sales_strata':int(rows.purchase_events.eq(0).sum()),
            'total_mode':total_mode,'arrival_prior_strength_days':prior.strength_days,
            'mu_prior_source':problem['mu_prior_source'],
            'no_observed_purchases':bool(rows.purchase_events.sum()==0),
            'inference_scope':'external_priors_and_total_only_no_observed_purchases' if rows.purchase_events.sum()==0 else 'conditional_on_offers_behavior_and_external_total',
            'fit_status':'joint_MAP_exact_total_likelihood' if optimum.success else 'optimizer_nonconvergence_approximate_only'})
    assumptions=dict(inputs.assumptions)
    assumptions.update({'estimation':'joint_MAP_observed_marked_events_and_exact_latent_count',
        'total_mode':total_mode,'expected_volume_not_forced_to_realized_total':True,
        'rates_prior':'Gamma_shape_1_plus_strength_times_mode_rate_strength',
        'point_estimate_uncertainty':'conditional_CP_variability_only_parameters_are_plugin_MAP',
        'mixture_components':len(_mixture(config)[0]),
        'budget_scale_semantics':'exp_mu_base_component_scale_at_DtD30_qty1_zero_cabin_offset',
        'median_amount_semantics':'mixture_median_at_DtD30_qty1_zero_cabin_offset',
        'no_purchase_flights':'external_quantity_and_budget_priors_required_not_identified_from_bookings',
        'policy_assumption':'given_offer_history_unrecorded_refusals_do_not_change_future_observed_policy',
        'amount_semantics':'assumed_per_seat_WTP_needs_confirmation'})
    return {'strata':pd.concat(output_rows,ignore_index=True),'price_segments':inputs.price_segments.copy(),
        'base_marks':pd.concat(base_parts,ignore_index=True),'marks':pd.concat(marks_parts,ignore_index=True),
        'posterior_marks':pd.concat(post_parts,ignore_index=True),'parameters':pd.DataFrame(parameter_rows),
        'group_distributions':full['group_distributions'],'assumptions':assumptions,'config':config,
        'primary_demand_seats':targets,'arrival_prior':prior}


def _draw_mixture_amount(location,config,price,mode,rng):
    w,o,s=_mixture(config)
    if mode=='unrestricted':
        j=int(rng.choice(len(w),p=w))
        return float(np.exp(location+o[j]+s[j]*rng.normal()))
    else:
        z=(log(float(price))-location-o)/s
        logmass=log_ndtr(z if mode=='below_price' else -z)
        logs=np.log(w)+logmass
        probabilities=np.exp(logs-logsumexp(logs))
    j=int(rng.choice(len(w),p=probabilities))
    # log-CDF inverse remains accurate for far-tail truncation. In particular,
    # above-price draws use the survival probability, never 1-CDF cancellation.
    logp=float(logmass[j])+log(max(float(rng.random()),np.nextafter(0.,1.)))
    normal=float(ndtri_exp(logp))
    normal=normal if mode=='below_price' else -normal
    value=float(np.exp(location+o[j]+s[j]*normal))
    return min(value,np.nextafter(float(price),0.)) if mode=='below_price' else max(value,float(price))


def sample_passengers_v3(inputs,result,seed=20261111):
    """Generate exact-total hidden journal using fitted ORIGINAL intensities.

    Do not feed posterior expected groups back as CP intensities: that would
    condition twice and change the law. This routine uses result['marks'], not
    posterior_marks, then chooses component/reason/time conditional on q.
    """
    rng=np.random.default_rng(seed);config=result['config']
    mus=result['parameters'].set_index('flight').mu_log.to_dict();offsets=config.cabin_offsets or {}
    records=[]
    for i,event in enumerate(inputs.bookings.sort_values(['flight','timestamp','source_row']).to_dict('records')):
        success=event['commercial']>0
        loc=mus[event['flight']]+config.beta_time*log((1+event['dtd'])/31)+offsets.get(event['cabin'],0)+config.beta_qty*log(event['qty'])
        amount=_draw_mixture_amount(loc,config,event['price'],'above_price',rng) if success else np.nan
        records.append({'timestamp':event['timestamp'],'id':f'REC-{event["flight"]}-{i:05d}',
            'flight':event['flight'],'cabin':event['cabin'],'rbd':event.get('rbd',pd.NA),'qty':event['qty'],
            'price':event['price'],'amount':amount,'outcome':'success' if success else 'cancel',
            'signed_qty':event['commercial'],'dtd':event['dtd'],'period':event['period'],
            'source_row':event['source_row'],'source_file':event.get('source_file',pd.NA),
            'cabin_source':'observed_booking','row_source':'observed_booking_operation','id_is_synthetic':True,
            'amount_source':'model_conditional_given_success' if success else 'not_available',
            'reason':'observed_purchase' if success else 'observed_cancel_unknown_link',
            'reason_is_modelled':False,'cancel_link_id':pd.NA})
    for flight,primary in result['primary_demand_seats'].items():
        group=result['marks'].loc[result['marks'].flight.eq(flight)&result['marks'].expected_hidden_groups.gt(0)]
        gross=int(inputs.purchases.loc[inputs.purchases.flight.eq(flight),'qty'].sum());hidden=int(primary)-gross
        quantities=_conditional_group_counts(group.groupby('qty').expected_hidden_groups.sum().to_dict(),hidden,rng)
        for i,q in enumerate(quantities):
            options=group.loc[group.qty.eq(q)];p=options.expected_hidden_groups.to_numpy(float,copy=True);p/=p.sum()
            chosen=options.iloc[int(rng.choice(len(options),p=p))]
            timestamp=chosen.start+pd.Timedelta(seconds=float(rng.random()*(chosen.end-chosen.start).total_seconds()))
            mode='unrestricted' if chosen.reason=='availability' else 'below_price' if chosen.reason=='budget' else 'above_price'
            amount=_draw_mixture_amount(float(chosen.wtp_log_location),config,chosen.price,mode,rng)
            records.append({'timestamp':timestamp,'id':f'SYN-V3-{flight}-{i:05d}',
                'flight':flight,'cabin':pd.NA,'model_preferred_cabin':chosen.cabin,
                'cabin_source':'not_observed_latent_model_label','rbd':pd.NA,'qty':int(q),'price':np.nan,
                'offered_price':chosen.price,'amount':amount,'outcome':'waste','signed_qty':0,
                'dtd':int(chosen.dtd),'period':chosen.period,'source_row':np.nan,'source_file':pd.NA,
                'row_source':'model_generated_hidden_group','id_is_synthetic':True,
                'amount_source':f'model_conditional_{chosen.reason}','reason':chosen.reason,
                'reason_is_modelled':True,'cancel_link_id':pd.NA})
    frame=pd.DataFrame(records)
    if frame.empty:
        frame=pd.DataFrame(columns=['flight','timestamp','id','cabin','qty','price','amount','outcome',
                                    'signed_qty','dtd','period','row_source','reason','reason_is_modelled'])
        frame['timestamp']=pd.to_datetime(frame.timestamp,utc=True)
    frame=frame.sort_values(['flight','timestamp','id']).reset_index(drop=True)
    for flight,target in result['primary_demand_seats'].items():
        if int(frame.loc[frame.flight.eq(flight)&frame.outcome.isin(['success','waste']),'qty'].sum())!=int(target):
            raise AssertionError('Conditional reconstruction violated exact primary-seat count.')
    frame.attrs['assumptions']=result['assumptions']
    return frame

#!/usr/bin/env python3
"""Reproducible boundary-detection benchmark. 

Requires: numpy scipy pandas scikit-learn jinja2. External methods are provided by the user
via --adapter MODULE; no purported Hickok/BEPS implementations are fabricated.

BEPS is ONE algorithm call (adapter function adjusted_angle), returning both its
initial and adjusted results. The benchmark evaluates initial and adjusted BoS
as TWO SCORE OUTPUTS of the same method, never as two separately run methods.

Implement the support boundary evaluation as well as the local-sector one and the distance to the boundary (depth).

"""
from __future__ import annotations
import argparse, importlib, json, math, traceback
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from distance_normalization import geometry_distance_scale, normalized_target_errors, TARGET_COLUMNS
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, balanced_accuracy_score

METHODS = ['DU', 'Hyperplane', 'Tangent Plane', 'BEPS', 'Oriented']
EXTERNAL = METHODS.copy()

@dataclass
class Case:
    name: str
    X: np.ndarray
    intrinsic_dim: int
    task: str                  # 'support' or 'local'
    d_boundary: np.ndarray | None = None
    gap: np.ndarray | None = None
    eval_mask: np.ndarray | None = None
    note: str = ''

def disk(n, rng):
    t=rng.uniform(0, 2*np.pi, n); r=np.sqrt(rng.uniform(0,1,n))
    X=np.c_[r*np.cos(t),r*np.sin(t)]
    return X, 1-r

def disk_nonuniform(n,rng):
    # Mixture of central and near-boundary radial distributions creates sharp
    # density variation while the geometric support remains the full unit disk.
    t=rng.uniform(0,2*np.pi,n)
    mix=rng.random(n)<.75
    r=np.where(mix,np.sqrt(rng.random(n))*.55,
               np.sqrt(.55**2+(1-.55**2)*rng.random(n)))
    return np.c_[r*np.cos(t),r*np.sin(t)],1-r

def annulus(n,rng):
    t=rng.uniform(0,2*np.pi,n); r=np.sqrt(rng.uniform(0,1,n)*3+1)
    return np.c_[r*np.cos(t),r*np.sin(t)], np.minimum(r-1,2-r)

def star(n,rng):
    # r(theta) = 1 + a*cos(m*theta); exact distance to boundary via dense
    # independent curve discretization (approximation error <= curve spacing).
    a,m=.32,5; t=rng.uniform(0,2*np.pi,n); r=np.sqrt(rng.random(n))*(1+a*np.cos(m*t))
    X=np.c_[r*np.cos(t),r*np.sin(t)]
    phi=np.linspace(0,2*np.pi,30000,endpoint=False)
    R=1+a*np.cos(m*phi); boundary=np.c_[R*np.cos(phi),R*np.sin(phi)]
    d=cKDTree(boundary).query(X)[0]
    return X,d

def crescent(n,rng):
    # Disk of radius 1.5 minus offset disk: both circular boundary arcs known.
    outer_R=1.5; center=np.array([.60,0.]); hole_R=1.15
    parts=[]; count=0
    while count<n:
        m=max(256,(n-count)*4)
        t=rng.uniform(0,2*np.pi,m); r=outer_R*np.sqrt(rng.random(m))
        Y=np.c_[r*np.cos(t),r*np.sin(t)]
        Y=Y[np.linalg.norm(Y-center,axis=1)>=hole_R]
        parts.append(Y);count+=len(Y)
    X=np.concatenate(parts)[:n]
    # Closest boundary point must lie on the valid arc of either circle.
    # Numerically approximate these arcs with very dense sampled points.
    t=np.linspace(0,2*np.pi,30000,endpoint=False)
    outer=outer_R*np.c_[np.cos(t),np.sin(t)]
    outer=outer[np.linalg.norm(outer-center,axis=1)>=hole_R-1e-10]
    inner=center+hole_R*np.c_[np.cos(t),np.sin(t)]
    inner=inner[np.linalg.norm(inner,axis=1)<=outer_R+1e-10]
    return X,cKDTree(np.vstack([outer,inner])).query(X)[0]

def ball(n,d,rng):
    Z=rng.normal(size=(n,d));Z/=np.linalg.norm(Z,axis=1,keepdims=True)
    r=rng.random(n)**(1/d);return Z*r[:,None],1-r

def shell(n,d,rng,inner=.65):
    Z=rng.normal(size=(n,d));Z/=np.linalg.norm(Z,axis=1,keepdims=True)
    r=(inner**d+(1-inner**d)*rng.random(n))**(1/d)
    return Z*r[:,None],np.minimum(r-inner,1-r)

def product_star(n,d,rng):
    A,da=star(n,rng)
    if d==2:return A,da
    B=rng.uniform(-1,1,size=(n,d-2))
    db=np.min(1-np.abs(B),axis=1)
    return np.c_[A,B],np.minimum(da,db)

def solid_torus(n,rng):
    # 3D tube of radius r around a major circle R; distance to tube surface.
    R,r=2.,.7; phi=rng.uniform(0,2*np.pi,n)
    theta=rng.uniform(0,2*np.pi,n);rho=r*np.sqrt(rng.random(n))
    X=np.c_[(R+rho*np.cos(theta))*np.cos(phi),(R+rho*np.cos(theta))*np.sin(phi),rho*np.sin(theta)]
    return X,r-rho

def swiss_roll(n,rng):
    # 2D manifold parameterized by (t,z) on a rectangle. Ground truth is
    # distance to manifold edges, approximated in intrinsic Euclidean arc length.
    t=rng.uniform(1.5*np.pi,4.5*np.pi,n);z=rng.uniform(-2,2,n)
    X=np.c_[t*np.cos(t), z, t*np.sin(t)]
    grid=np.linspace(1.5*np.pi,4.5*np.pi,30000)
    arc=np.r_[0,np.cumsum(.5*(np.sqrt(1+grid[:-1]**2)+np.sqrt(1+grid[1:]**2))*np.diff(grid))]
    st=np.interp(t,grid,arc)
    dist=np.minimum.reduce([st,arc[-1]-st,z+2,2-z])
    return X,dist

def embed(X,D,rng,noise=0):
    d=X.shape[1]
    if D<d:raise ValueError('ambient dimension must >= intrinsic')
    if D==d: return X.copy() + noise*rng.normal(size=X.shape)
    Q,_=np.linalg.qr(rng.normal(size=(D,d)))
    Y=X@Q.T
    if noise:Y+=noise*rng.normal(size=Y.shape)
    return Y

def support_cases(n,rng,dimensions,noise):
    spec=[('disk',2,lambda:disk(n,rng)),('annulus',2,lambda:annulus(n,rng)),
          ('disk_nonuniform',2,lambda:disk_nonuniform(n,rng)),
          ('star',2,lambda:star(n,rng)),('crescent',2,lambda:crescent(n,rng)),
          ('torus',3,lambda:solid_torus(n,rng)),('swiss_roll',2,lambda:swiss_roll(n,rng))]
    for d in (3,5,10):
        spec += [(f'ball_d{d}',d,lambda d=d:ball(n,d,rng)),
                 (f'shell_d{d}',d,lambda d=d:shell(n,d,rng))]
    for d in (5,10):spec.append((f'star_product_d{d}',d,lambda d=d:product_star(n,d,rng)))
    for name,d,gen in spec:
        X,dist=gen()
        base_dim=X.shape[1]
        targets=sorted(set([base_dim]+[D for D in dimensions if D>=base_dim]))
        for D in targets:
            Y=embed(X,D,rng,noise)
            yield Case(f'{name}_D{D}',Y,d,'support',d_boundary=dist,
                note=('inherited latent boundary (noisy observations)' if noise else 'known domain boundary'))

def local_sector_cases(k, n_patch, intrinsic_dimensions, ambient_dimensions,rng,noise):
    """Independent focal neighborhoods, with KNOWN occupied spherical cap.

    For d=2, occupied directions have exact angle alpha and empty gap 2pi-alpha.
    For d>2, we sample unit directions from a spherical cap of half-angle phi;
    cap EMPTY SOLID-ANGLE FRACTION is (1-cdf_cos(phi)) approximately determined
    by an empirical distribution of uniform sphere directions, via closed beta CDF.
    This is a measure of directional exclusion, NOT global support boundary.

    Each patch is spatially isolated in its own cluster; only the focal point is
    evaluated. With k satellite points, their distances are all < patch separation.
    """
    from scipy.special import betainc
    degrees=[60,90,120,180,240,300,360]
    for d in intrinsic_dimensions:
        if d<2:continue
        for D in sorted(set([d]+[a for a in ambient_dimensions if a>=d])):
            focal=[]; satellites=[]; gaps=[]; indices=[]
            # Centers separated along coordinate 1; no cross-patch kNN for moderate D.
            for p in range(n_patch):
                angle=degrees[p%len(degrees)]
                if d==2:
                    theta=rng.uniform(-angle/2,angle/2,k)*np.pi/180
                    vec=np.c_[np.cos(theta),np.sin(theta)]
                    empty=1-angle/360
                else:
                    # cap half-angle angle/2, angular aperture max 180deg
                    phi=np.deg2rad(angle/2)
                    # rejection from uniform sphere (may be inefficient for small caps
                    # at high d); directly sample polar angle via truncated beta CDF.
                    # radial polar distribution sin(theta)^(d-2), CDF by betainc.
                    from scipy.special import betaincinv
                    cap_p= .5*betainc((d-1)/2,.5,np.sin(phi)**2) if phi<=np.pi/2 else 1-.5*betainc((d-1)/2,.5,np.sin(phi)**2)
                    u=rng.uniform(0,cap_p,k)
                    # inverse polar CDF by root/bisection (efficient in small batches)
                    from scipy.optimize import brentq
                    def cdf(th):
                        z=.5*betainc((d-1)/2,.5,np.sin(th)**2)
                        return z if th<=np.pi/2 else 1-z
                    ang=np.array([brentq(lambda z:cdf(z)-v,0,np.pi) for v in u])
                    orth=rng.normal(size=(k,d-1));orth/=np.linalg.norm(orth,axis=1,keepdims=True)
                    vec=np.c_[np.cos(ang),np.sin(ang)[:,None]*orth]
                    empty=1-cap_p
                # Choose a different orientation per patch via random orthogonal transform
                Q,_=np.linalg.qr(rng.normal(size=(d,d)))
                vec=vec@Q
                radius=rng.uniform(.2,.9,k)
                center=np.zeros(d);center[0]=10*p
                indices.append(len(focal)*(k+1))
                focal.append(np.vstack([center,center+radius[:,None]*vec]))
                gaps.append(empty)
            X=np.concatenate(focal)
            Y=embed(X,D,rng,noise)
            mask=np.zeros(len(Y),dtype=bool);mask[np.array(indices)]=True
            gap=np.full(len(Y),np.nan);gap[mask]=gaps
            yield Case(f'local_sector_d{d}_D{D}',Y,d,'local',gap=gap,eval_mask=mask,
                note='Known population missing directional fraction; only patch centers scored')


def methods_from_adapter(module):
    if not module:return {}
    usermod=importlib.import_module(module)
    return {name: getattr(usermod, {'DU':'du','Hyperplane':'hyperplane',
     'Tangent Plane':'tangent_plane','BEPS':'adjusted_angle',
     'Oriented':'oriented'}[name])
      for name in EXTERNAL if hasattr(usermod, {'DU':'du','Hyperplane':'hyperplane',
     'Tangent Plane':'tangent_plane','BEPS':'adjusted_angle',
     'Oriented':'oriented'}[name])}


def combined_beps_outputs(result, n):
    """Parse the TWO stages computed by ONE adjusted_angle() call.

    Mandatory length-n channels:
      initial_bos, initial_angle_deg, adjusted_bos, adjusted_angle_deg

    Final outputs ALWAYS fall back to the initial outputs when the adjustment
    is not applied. In particular, an unchanged, non-extreme adjusted BoS may
    have a perfectly valid final angle. Do NOT replace that angle with NaN.

    Optional 'adjustment_applied': boolean length-n mask indicating whether the
    adjustment step fired. If absent, infer a *numeric change* from either the
    angle or the BoS, using tight tolerances. Numeric change is a proxy, not a
    logical guarantee: an executed adjustment might theoretically leave both
    outputs unchanged. Pass the explicit mask if your algorithm exposes it.

    Returns the four numeric channels PLUS 'adjustment_applied' (bool).
    """
    keys=('initial_bos','initial_angle_deg','adjusted_bos','adjusted_angle_deg')
    if not isinstance(result, dict):
        raise TypeError('adjusted_angle() must return a dict with '+', '.join(keys))
    missing=[key for key in keys if key not in result]
    if missing:
        raise ValueError('adjusted_angle() missing output(s): '+', '.join(missing))
    arrays={}
    for key in keys:
        arr=np.asarray(result[key],dtype=float)
        if arr.shape!=(n,):
            raise ValueError(f'{key}: expected shape {(n,)}, got {arr.shape}')
        arrays[key]=arr
    for key in ('initial_bos','adjusted_bos'):
        a=arrays[key]
        if np.any(np.isfinite(a)&((a<-1e-9)|(a>1+1e-9))):
            raise ValueError(f'{key} contains values outside the BoS range [0,1]')

    bos_changed=~np.isclose(arrays['initial_bos'],arrays['adjusted_bos'],
                            atol=1e-8,rtol=1e-9,equal_nan=True)
    angle_changed=~np.isclose(arrays['initial_angle_deg'],arrays['adjusted_angle_deg'],
                              atol=1e-7,rtol=1e-9,equal_nan=True)
    inferred_changed=bos_changed|angle_changed
    if 'adjustment_applied' in result and result['adjustment_applied'] is not None:
        explicit=np.asarray(result['adjustment_applied'])
        if explicit.shape!=(n,):
            raise ValueError(f'adjustment_applied: expected shape {(n,)}, got {explicit.shape}')
        if not np.issubdtype(explicit.dtype,np.bool_):
            raise TypeError('adjustment_applied must be a boolean array')
        if np.any(inferred_changed & ~explicit):
            raise ValueError('adjustment_applied=False contradicts changed BEPS outputs')
        arrays['adjustment_applied']=explicit.copy()
    else:
        arrays['adjustment_applied']=inferred_changed
    return arrays

def optional_oriented_outputs(result, n):
    """Parse optional ORIENTED-specific outputs from an adapter.

    The mandatory 'scores' channel is handled by scalar_scores().  Additional
    keys are optional and must be length n:
       'angle_deg'         oriented cone aperture in degrees, NaN if undefined
       'raw_bos'           raw oriented BoS (if scores are transformed)
       'distance_estimate' predicted geometric distance, in the SAME units as X

    Never silently interpret an aperture in degrees as Euclidean distance.
    """
    outputs={}
    if not isinstance(result, dict):return outputs
    for key in ('angle_deg', 'raw_bos', 'distance_estimate'):
        if key in result and result[key] is not None:
            v=np.asarray(result[key], dtype=float)
            if v.shape != (n,):
                raise ValueError(f"oriented {key}: expected {(n,)}, got {v.shape}")
            outputs[key]=v
    if 'raw_bos' not in outputs and result.get('type')=='bos':
        outputs['raw_bos']=np.asarray(result['scores'],float)
    return outputs


def _safe_spearman(a, b):
    """Return NaN for insufficient or constant data rather than warnings."""
    valid=np.isfinite(a)&np.isfinite(b)
    if np.count_nonzero(valid)<4:return np.nan
    a,b=np.asarray(a)[valid],np.asarray(b)[valid]
    if np.ptp(a)==0 or np.ptp(b)==0:return np.nan
    return float(spearmanr(a,b).statistic)


def evaluate_oriented_distance(case, scores, extra):
    """Distance validation on ALL support points, not only binary margin points.

    This is a separate scientific claim from near-boundary classification.
    Spearman signs are PREDECLARED rather than flipped using test labels:
      score vs -distance: positive supports 'higher score = nearer boundary'
      raw BoS vs -distance: positive supports 'BoS larger = nearer boundary'
      angle vs distance: signed, no a priori monotonic direction assumed

    If the adapter has an explicit geometric-unit 'distance_estimate', evaluate
    it directly. Otherwise report no MAE/RMSE: angle is NOT distance.
    """
    if case.task!='support':return None
    truth=np.asarray(case.d_boundary,dtype=float)
    s=np.asarray(scores,dtype=float)
    full=np.isfinite(truth)&np.isfinite(s)
    valid_score=int(full.sum())
    row={
        'Dataset':case.name, 'n_total':len(truth), 'n_score':valid_score,
        'ScoreCoverage':valid_score/len(truth),
        'ScoreRhoVsNegDistance':_safe_spearman(s,-truth),
        'AngleCoverage':np.nan, 'n_angle':0,
        'AngleRhoVsDistance':np.nan,
        'AngleRhoVsNegDistance':np.nan,
        'RawBoSRhoVsNegDistance':np.nan,
        'DistanceEstimateCoverage':np.nan,
        'DistanceMAE':np.nan,'DistanceRMSE':np.nan,
        'DistanceNormMAE':np.nan, 'DistanceNormRMSE':np.nan,
        'MedianTruthDistance':float(np.median(truth[np.isfinite(truth)])),
    }
    a=extra.get('angle_deg')
    if a is not None:
        va=np.isfinite(a)&np.isfinite(truth)
        row['n_angle']=int(va.sum())
        row['AngleCoverage']=float(va.mean())
        row['AngleRhoVsDistance']=_safe_spearman(a,truth)
        row['AngleRhoVsNegDistance']=_safe_spearman(a,-truth)
    raw=extra.get('raw_bos')
    if raw is not None:
        row['RawBoSRhoVsNegDistance']=_safe_spearman(raw,-truth)
    predicted=extra.get('distance_estimate')
    if predicted is not None:
        v=np.isfinite(predicted)&np.isfinite(truth)
        row['DistanceEstimateCoverage']=float(v.mean())
        if v.any():
            error=predicted[v]-truth[v]
            row['DistanceMAE']=float(np.mean(np.abs(error)))
            row['DistanceRMSE']=float(np.sqrt(np.mean(error**2)))
            # Normalize by mean boundary distance for scale-aware across-case
            # comparisons. Geometric-unit metrics above remain authoritative.
            scale=float(np.mean(truth[v]))
            if scale>0:
                row['DistanceNormMAE']=row['DistanceMAE']/scale
                row['DistanceNormRMSE']=row['DistanceRMSE']/scale
        geometric_scale=geometry_distance_scale(case.name)
        if geometric_scale is not None:
            _,target_errors=normalized_target_errors(truth,predicted,geometric_scale)
            row.update(target_errors)
    return row


def oriented_distance_binned(case, scores, angle_deg, n_bins=10):
    """Distance-decile summary for plots (no fitted relationship assumed)."""
    if case.task!='support':return []
    d=np.asarray(case.d_boundary,dtype=float)
    ok=np.isfinite(d)&np.isfinite(scores)
    if ok.sum()<n_bins:return []
    frame=pd.DataFrame({'d':d[ok],'score':np.asarray(scores)[ok]})
    frame['angle_deg']=angle_deg[ok] if angle_deg is not None else np.nan
    frame['bin']=pd.qcut(frame.d,q=n_bins,labels=False,duplicates='drop')
    rows=[]
    for i,g in frame.groupby('bin'):
        rows.append({'Dataset':case.name,'DistanceBin':int(i),'Count':len(g),
                     'MedianDistance':float(g.d.median()),
                     'MedianBoundaryScore':float(g.score.median()),
                     'MedianAngleDeg':float(g.angle_deg.median()) if g.angle_deg.notna().any() else np.nan,
                     'AngleCoverageInBin':float(g.angle_deg.notna().mean())})
    return rows


def evaluate_beps_angles(case, outputs, eps, deep, local_threshold):
    """Quantify BEPS initial/final angles AND frequency of genuine adjustment.

    Unlike earlier versions, final angles include the unmodified fallback
    values. Thus final-angle *availability* is NOT the adjustment rate. A
    separate flag marks changed outputs (or a user-supplied adjustment mask).
    Report correlations for all valid outputs and, separately, on points
    where an adjustment was applied. Correlations are descriptive, not a
    direct error measurement against true empty-sector aperture.
    """
    if case.task=='local':
        mask=np.asarray(case.eval_mask,dtype=bool)
        truth=np.asarray(case.gap,float)
        positive=mask & (truth>=local_threshold)
        negative=mask & (truth<local_threshold)
        target='MissingDirectionalFraction'
    else:
        truth=np.asarray(case.d_boundary,float)
        mask=np.isfinite(truth)
        positive=mask & (truth<=eps)
        negative=mask & (truth>=deep)
        target='TrueBoundaryDistance'
    ini=np.asarray(outputs['initial_angle_deg'],float)
    adj=np.asarray(outputs['adjusted_angle_deg'],float)
    changed=np.asarray(outputs['adjustment_applied'],bool)
    modified=mask & changed
    finite_shift=modified & np.isfinite(ini) & np.isfinite(adj)
    n_pos=int(positive.sum());n_neg=int(negative.sum())
    return {
        'Dataset':case.name, 'Task':case.task,
        'AngleTarget':target,
        'n_evaluable':int(mask.sum()),
        'n_positive':n_pos, 'n_negative':n_neg,
        'InitialAngleCoverage':float(np.isfinite(ini[mask]).mean()) if mask.any() else np.nan,
        'FinalAngleCoverage':float(np.isfinite(adj[mask]).mean()) if mask.any() else np.nan,
        'AdjustmentRate':float(changed[mask].mean()) if mask.any() else np.nan,
        'AdjustmentRateOnPositives':float(changed[positive].mean()) if n_pos else np.nan,
        'AdjustmentRateOnNegatives':float(changed[negative].mean()) if n_neg else np.nan,
        'InitialAngleRhoVsTarget':_safe_spearman(ini[mask],truth[mask]),
        'FinalAngleRhoVsTarget':_safe_spearman(adj[mask],truth[mask]),
        'FinalAngleRhoVsTargetOnAdjusted':_safe_spearman(adj[modified],truth[modified]),
        'MeanAbsAngleShiftDegOnAdjusted':float(np.mean(np.abs(adj[finite_shift]-ini[finite_shift]))) if finite_shift.any() else np.nan,
        'n_adjustments':int(modified.sum()),
        'n_valid_angle_shifts':int(finite_shift.sum()),
    }

def scalar_scores(result,n):
    # Adapter must return a 1D vector or dict with 'scores'; all scores are
    # required to be high=boundary, except BoS values optionally flagged.
    if isinstance(result,dict):
        v=np.asarray(result['scores'],float)
        if result.get('type')=='bos':v=2*np.abs(v-.5)
        if result.get('direction')=='low':v=-v
    else:v=np.asarray(result,float)
    if v.shape!=(n,):raise ValueError(f'expected shape {(n,)}, got {v.shape}')
    return v

def evaluate(case,method,scores,eps,deep,local_threshold,all_point_spearman=False):
    if case.task=='support':
        d=case.d_boundary
        # Near-boundary vs deep interior; exclude ambiguous transition zone.
        valid=np.isfinite(scores)&((d<=eps)|(d>=deep))
        y=(d<=eps).astype(int)
    else:
        valid=np.isfinite(scores)&case.eval_mask
        y=(case.gap>=local_threshold).astype(int)
    n_positive=int(np.count_nonzero(y[valid]))
    counts={'n_positive':n_positive,'n_negative':int(valid.sum())-n_positive}
    corr_valid=np.isfinite(scores) if all_point_spearman else valid
    corr_target=case.d_boundary[corr_valid] if case.task=='support' else case.gap[corr_valid]
    rho=_safe_spearman(scores[corr_valid],corr_target)
    if case.task=='support':rho=-rho
    if valid.sum()<4 or len(np.unique(y[valid]))<2:
        return {'AUROC':np.nan,'AUPRC':np.nan,'BalancedAcc':np.nan,'F1':np.nan,
                'Spearman':rho,'n_eval':int(valid.sum()),**counts}
    target=y[valid];s=scores[valid]
    try:auc=roc_auc_score(target,s);ap=average_precision_score(target,s)
    except ValueError:auc=ap=np.nan
    # Fair fixed-prevalence threshold (top n_positive predictions) is an
    # *oracle prevalence* operating point. NOT a deployable trained threshold.
    pred=np.zeros(len(s),dtype=int)
    pred[np.argsort(s)[-int(target.sum()):]]=1
    _,_,f1,_=precision_recall_fscore_support(target,pred,average='binary',zero_division=0)
    return {'AUROC':auc,'AUPRC':ap,'BalancedAcc':balanced_accuracy_score(target,pred),
           'F1':f1,'Spearman':rho,'n_eval':int(valid.sum()),**counts}

def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    adapters=methods_from_adapter(args.adapter)
    dims=tuple(sorted(set(int(v) for v in args.ambient.split(','))))
    intr=tuple(sorted(set(int(v) for v in args.local_intrinsic.split(','))))
    task=getattr(args,'task','all')
    # Consume the original RNG sequence even for local-only runs, preserving
    # the exact neighborhoods generated after support cases in a full run.
    cases=[]
    for case in support_cases(args.n,rng,dims,args.noise):
        if task in ('all','support'):cases.append(case)
    if task in ('all','local'):
        cases+=list(local_sector_cases(args.k,args.patches,intr,dims,rng,args.noise))
    resume=getattr(args,'resume_relaxed',False)
    previous={}
    targets=set()
    if resume:
        if task!='all':raise ValueError('--resume-relaxed requires --task all')
        metrics_path=out/'metrics.csv'
        if not metrics_path.exists():
            raise ValueError('--resume-relaxed requires an existing metrics.csv in --out')
        old=pd.read_csv(metrics_path,keep_default_na=False,na_values=[''])
        missing=pd.to_numeric(old.AUROC,errors='coerce').isna()
        targets=set(old.loc[old.Task.eq('support') & old.Status.eq('OK') & missing,'Dataset'])
        if not targets:
            print('No support datasets with undefined AUROC need repair.');return
        needed=set(old.loc[old.Dataset.isin(targets) & old.Status.eq('OK'),'Method'])
        if needed-set(adapters):
            raise ValueError('Repair requires adapters for: '+', '.join(sorted(needed-set(adapters))))
        # Refuse a different regeneration configuration before running methods.
        generated={case.name:case for case in cases}
        for name in old.Dataset.unique():
            if name not in generated:
                raise ValueError('Dataset configuration differs: '+name)
        manifest_path=out/'dataset_manifest.csv'
        if manifest_path.exists():
            prior_manifest=pd.read_csv(manifest_path)
            for row in prior_manifest.itertuples():
                if len(generated[row.dataset].X)!=row.n:
                    raise ValueError('Use the original --n and --patches settings')
        original_rows=old.loc[old.Task.eq('support') & ~old.Details.fillna('').str.startswith('Relaxed labels:')]
        for name in original_rows.Dataset.unique():
            case=generated[name]
            kth=cKDTree(case.X).query(case.X,k=min(args.k+1,len(case.X)))[0][:,-1]
            expected=args.eps_factor*float(np.median(kth))
            recorded=float(original_rows.loc[original_rows.Dataset.eq(name),'eps'].iloc[0])
            if not np.isclose(expected,recorded,rtol=1e-9,atol=1e-12):
                raise ValueError('Regenerated geometry/threshold differs for '+name+
                                 '; supply original --k, --n, --seed, --noise and --ambient')
        for name in ('metrics.csv','dataset_manifest.csv','oriented_distance_metrics.csv',
                     'oriented_distance_bins.csv','beps_angular_metrics.csv'):
            path=out/name
            if path.exists():
                previous[name]=pd.read_csv(path,keep_default_na=False,na_values=[''])
                backup=out/(name+'.before_relaxed')
                if not backup.exists():backup.write_bytes(path.read_bytes())
        cases=[case for case in cases if case.name in targets]
        print('Repairing '+str(len(cases))+' datasets with percentile labels',flush=True)
    def write_csv(frame,name):
        if name in previous:
            prior=previous[name]
            key='dataset' if name=='dataset_manifest.csv' else 'Dataset'
            frame=pd.concat([prior.loc[~prior[key].isin(targets)],frame],ignore_index=True)
        frame.to_csv(out/name,index=False)
        return frame
    results=[]; errors=[];manifest=[]
    oriented_results=[]; oriented_bins=[];beps_angle_results=[]
    for case in cases:
        case.X=np.asarray(case.X,dtype=float)
        manifest.append({'dataset':case.name,'task':case.task,'n':len(case.X),
                         'intrinsic_dim':case.intrinsic_dim,'ambient_dim':case.X.shape[1],
                         'note':case.note})
        if args.save_datasets:
            np.savez_compressed(out/(case.name+'.npz'),X=case.X,
                d_boundary=case.d_boundary if case.d_boundary is not None else np.array([]),
                gap=case.gap if case.gap is not None else np.array([]),
                eval_mask=case.eval_mask if case.eval_mask is not None else np.array([]))
        # Per-dataset thresholds relative to geometric scale. Near=eps*kNN median.
        # This makes ground truth resolution-aware but remains independent of scores.
        if case.task=='support':
            kth=cKDTree(case.X).query(case.X,k=min(args.k+1,len(case.X)))[0][:,-1]
            eps=args.eps_factor*float(np.median(kth))
            deep=max(args.deep_factor*eps,eps+1e-12)
        else:eps=deep=np.nan
        relaxed=resume and case.task=='support'
        if relaxed:
            eps,deep=np.quantile(case.d_boundary,[.2,.5])
            if eps>=deep:
                raise ValueError('Distance quantiles cannot separate classes for '+case.name)
        saved_scores={}
        for method in METHODS:
            status='OK';details=('Relaxed labels: nearest 20% vs farthest 50%; Spearman on all finite scores' if relaxed else '')
            try:
                if method=='BEPS':
                    if method not in adapters:raise NotImplementedError('combined adjusted_angle() adapter not supplied')
                    # Important: ONE invocation per dataset; both score channels
                    # and both angle channels come from THIS SAME computation.
                    beps_options={}
                    if getattr(args,'prop_thresh',None) is not None:
                        beps_options['prop_thresh']=args.prop_thresh
                    combined=combined_beps_outputs(
                        adapters[method](case.X,k=args.k,intrinsic_dim=case.intrinsic_dim,**beps_options),
                        len(case.X))
                    if args.save_datasets:
                        np.savez_compressed(out/(case.name+'_beps_outputs.npz'),**combined)
                    beps_angle_results.append(evaluate_beps_angles(
                        case,combined,eps,deep,args.local_threshold))
                    for label,key in (('Initial BoS','initial_bos'),
                                      ('Adjusted BoS','adjusted_bos')):
                        # Both are raw BoS in [0,1]; use two-sided boundary evidence.
                        scores=2*np.abs(combined[key]-.5)
                        saved_scores[key]=combined[key]
                        saved_scores[key+'_boundary_score']=scores
                        m=evaluate(case,method,scores,eps,deep,args.local_threshold,all_point_spearman=relaxed)
                        results.append({'Dataset':case.name,'Task':case.task,
                           'Method':method,'Output':label,'Status':status,
                           'eps':eps,'deep_cutoff':deep,'Details':details,**m})
                    continue
                if method not in adapters:raise NotImplementedError('adapter not supplied')
                result=adapters[method](case.X, k=args.k, intrinsic_dim=case.intrinsic_dim)
                scores=scalar_scores(result,len(case.X))
                if method=='Oriented' and case.task=='support':
                    extra=optional_oriented_outputs(result,len(case.X))
                    oriented_results.append(evaluate_oriented_distance(case,scores,extra))
                    oriented_bins.extend(oriented_distance_binned(
                        case,scores,extra.get('angle_deg'),n_bins=args.distance_bins))
                m=evaluate(case,method,scores,eps,deep,args.local_threshold,all_point_spearman=relaxed)
                saved_scores[method.replace(' ','_')+'_boundary_score']=scores
                if method=='Oriented' and isinstance(result,dict):
                    saved_scores['oriented_raw_bos']=np.asarray(result['scores'],float)
            except NotImplementedError as exc:
                status='N/A';details=str(exc);m={k:np.nan for k in ['AUROC','AUPRC','BalancedAcc','F1','Spearman']};m['n_eval']=0
            except Exception as exc:
                status='ERROR';details=str(exc)
                errors.append({'dataset':case.name,'method':method,'error':traceback.format_exc()})
                m={k:np.nan for k in ['AUROC','AUPRC','BalancedAcc','F1','Spearman']};m['n_eval']=0
            # If BEPS is unavailable, record BOTH missing outputs explicitly.
            outputs=('Initial BoS','Adjusted BoS') if method=='BEPS' else ('Boundary score',)
            for label in outputs:
                results.append({'Dataset':case.name,'Task':case.task,
                    'Method':method,'Output':label,'Status':status,
                    'eps':eps,'deep_cutoff':deep,'Details':details,**m})
        if args.save_datasets:
            np.savez_compressed(out/(case.name+'_method_scores.npz'),**saved_scores)
        print(f'{case.name}: complete',flush=True)
    df=write_csv(pd.DataFrame(results),'metrics.csv')
    write_csv(pd.DataFrame(manifest),'dataset_manifest.csv')
    # A missing adapter does NOT imply zero distance performance. Emit a clean
    # header-only CSV and an explanatory LaTeX table when oriented is absent.
    oriented_columns=['Dataset','n_total','n_score','ScoreCoverage',
        'ScoreRhoVsNegDistance','n_angle','AngleCoverage','AngleRhoVsDistance',
        'AngleRhoVsNegDistance','RawBoSRhoVsNegDistance',
        'DistanceEstimateCoverage','DistanceMAE','DistanceRMSE',
        'DistanceNormMAE','DistanceNormRMSE','MedianTruthDistance']+TARGET_COLUMNS
    oriented_df=pd.DataFrame(oriented_results,columns=oriented_columns)
    oriented_df=write_csv(oriented_df,'oriented_distance_metrics.csv')
    write_csv(pd.DataFrame(oriented_bins,columns=['Dataset','DistanceBin','Count','MedianDistance',
        'MedianBoundaryScore','MedianAngleDeg','AngleCoverageInBin']),
            'oriented_distance_bins.csv')
    if len(oriented_df):
        display=oriented_df[['Dataset','ScoreRhoVsNegDistance','RawBoSRhoVsNegDistance',
           'AngleRhoVsDistance','AngleCoverage','DistanceMAE','DistanceRMSE',
           'DistanceTargetNormMAE','DistanceTargetNormRMSE']].copy()
        for c in display.columns[1:]:
            display[c]=display[c].map(lambda x:f'{x:.3f}' if pd.notna(x) else '--')
        (out/'oriented_distance_metrics.tex').write_text(display.to_latex(
           index=False,escape=True,longtable=True,
           caption='Oriented variant versus true support-boundary distance. '
           'TargetNorm errors compare the estimate with truth divided by maximum geometric depth. '
           'Angle correlations are signed; distance errors are available only '
           'when the adapter supplies geometric-unit predictions.',
           label='tab:oriented_distance_benchmark'))
    else:
        (out/'oriented_distance_metrics.tex').write_text(
            '% Oriented distance metrics unavailable: connect oriented() in the adapter.\n')
    beps_columns=['Dataset','Task','AngleTarget','n_evaluable','n_positive',
        'n_negative','InitialAngleCoverage','FinalAngleCoverage',
        'AdjustmentRate','AdjustmentRateOnPositives','AdjustmentRateOnNegatives',
        'InitialAngleRhoVsTarget','FinalAngleRhoVsTarget',
        'FinalAngleRhoVsTargetOnAdjusted','MeanAbsAngleShiftDegOnAdjusted',
        'n_adjustments','n_valid_angle_shifts']
    beps_angles_df=pd.DataFrame(beps_angle_results,columns=beps_columns)
    beps_angles_df=write_csv(beps_angles_df,'beps_angular_metrics.csv')
    if not beps_angles_df.empty:
        show=beps_angles_df[['Dataset','Task','InitialAngleCoverage',
            'FinalAngleCoverage','AdjustmentRate',
            'AdjustmentRateOnPositives','AdjustmentRateOnNegatives',
            'InitialAngleRhoVsTarget','FinalAngleRhoVsTarget',
            'MeanAbsAngleShiftDegOnAdjusted']].copy()
        for col in show.columns[2:]:
            show[col]=show[col].map(lambda x:f'{x:.3f}' if pd.notna(x) else '--')
        (out/'beps_angular_metrics.tex').write_text(show.to_latex(
            index=False,escape=True,longtable=True,
            caption='Angular outputs from ONE combined BEPS call. Final cone '
            'angles fall back to initial angles when no adjustment occurs. '
            'Adjustment rates are distinct from finite-angle coverage. '
            'Correlations are descriptive and not angular-estimation errors.',
            label='tab:beps_angular_benchmark'))
    else:
        (out/'beps_angular_metrics.tex').write_text(
            '% BEPS angular metrics unavailable: connect adjusted_angle() in the adapter.\n')
    if resume and (out/'errors.json').exists():
        prior_errors=json.loads((out/'errors.json').read_text())
        errors=[e for e in prior_errors if e.get('dataset') not in targets]+errors
    (out/'errors.json').write_text(json.dumps(errors,indent=2))
    # Tables with N/A (never impute uncomputed results as zero).
    for task in ['support','local']:
        t=df[df.Task==task].copy()
        for col in ['AUROC','AUPRC','BalancedAcc','F1','Spearman']:
            t[col]=t[col].map(lambda x:f'{x:.3f}' if pd.notna(x) else '--')
        t['Method']=pd.Categorical(t.Method,categories=METHODS,ordered=True)
        t=t.sort_values(['Dataset','Method','Output'])
        columns=['Dataset','Method','Output','AUROC','AUPRC','BalancedAcc','F1','Spearman']
        tex=t[columns].to_latex(index=False,escape=True,longtable=True,caption=f'{task.capitalize()}-boundary benchmark (one random seed). -- = unavailable or undefined.',label=f'tab:{task}_benchmark')
        (out/f'{task}_metrics.tex').write_text(tex)
    (out/('resume_relaxed_config.json' if resume else 'run_config.json')).write_text(json.dumps(vars(args),indent=2))
    print('Files written to:',out.resolve())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--adapter',help='Python module containing the five external algorithms: du, hyperplane, tangent_plane, adjusted_angle (combined BEPS), oriented')
    p.add_argument('--out',default='benchmark_results')
    p.add_argument('--task',choices=['all','support','local'],default='all',help='Evaluate only this task; local-only retains the full-run RNG sequence')
    p.add_argument('--prop-thresh',type=float,default=None,help='Override adjusted_angle adapter proportion threshold; otherwise retain its default')
    p.add_argument('--n',type=int,default=600)
    p.add_argument('--k',type=int,default=32)
    p.add_argument('--patches',type=int,default=70)
    p.add_argument('--ambient',default='2,3,10,50')
    p.add_argument('--local-intrinsic',default='2,3,5')
    p.add_argument('--noise',type=float,default=0.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eps-factor',type=float,default=.6,help='near-boundary threshold / median kth-NN distance')
    p.add_argument('--deep-factor',type=float,default=3.,help='deep-interior threshold / eps')
    p.add_argument('--local-threshold',type=float,default=.5,help='positive iff missing directional fraction >= threshold')
    p.add_argument('--distance-bins',type=int,default=10,help='Number of distance quantile bins for oriented diagnostic curves')
    p.add_argument('--save-datasets',action='store_true')
    p.add_argument('--resume-relaxed',action='store_true',help='Rerun only existing support datasets with OK status but undefined AUROC, using 20/50 distance quantiles. Supply original run settings.')
    run(p.parse_args())

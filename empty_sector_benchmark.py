#!/usr/bin/env python3
"""Controlled empty-sector and concavity benchmark for the combined BEPS algorithm.

Ground truth (2D ONLY):
  Sector patch: k neighbors are uniform in a known angular interval of width
  alpha around a focal point. Population empty angular sector = 360 - alpha.
    alpha < 180  -> convex wedge at focal point
    alpha = 180  -> flat/straight half-plane (excluded from concavity accuracy)
    180 < alpha < 360 -> concave (re-entrant) wedge at focal point
    alpha = 360 -> interior/full angular support (excluded from concavity)
  Annulus patch: neighbors uniformly sample the intersection of a radius-h
  ball around a focal point with the annulus 1 <= ||x|| <= 2.
    outer boundary r=2 -> convex; inner boundary r=1 -> concave.
  The annulus does NOT have a comparable fixed-degree empty-sector target;
  it is used ONLY to evaluate concavity/convexity predictions.

For every focal point, call ONE user-supplied `adjusted_angle()` implementation,
returning both initial and final BoS and angles (see v4 adapter contract).
The current published experiment uses --gap-policy score_bands:
  final BoS >= 0.7 -> convex boundary, estimated gap = 360 - final angle
  final BoS <= 0.3 -> concave boundary, estimated gap = final angle
  0.3 < final BoS < 0.7 -> interior, estimated gap = 0
Endpoints count as boundary. Predictions use the final score only; ground-truth
labels are used afterward for evaluation, never for deciding whether to assign
zero gap. No points are removed by this classification.
The command-line defaults use this final score-band experiment. Historical
evaluation helpers are retained only for compatibility tests; the CLI does not
offer them and no saved result uses them.

Crucial: the angle has a FALLBACK to initial_angle when no adjustment occurs.
Never infer an adjustment merely because an angle is finite. The score-band
classification applies to unchanged fallbacks too. There is no adjustment-execution filter.
All saved results and the default evaluation helper use score_bands.

Requirements: numpy, pandas, jinja2; the connected adapter also needs scikit-learn.
Example:
  python empty_sector_benchmark.py --adapter my_methods_adapter_TEMPLATE \
    --k 12,32,52 --repeats 20 --annulus-repeats 20 --prop-thresh 0.51 \
    --angular-gap 30 --gap-policy score_bands --out results/empty_sector

The default adapter connects to the bundled implementation in ../code.
"""
from __future__ import annotations
import argparse
import json
import inspect
from functools import partial
import importlib
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

# The flat/full cases check for false positives but are not labeled convex/concave.
DEFAULT_ANGLES = (60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360)
ATOL_BOS = 1e-8


def _load_adapter(spec: str):
    """Import an installed module or a Python-file path; use its adjusted_angle()."""
    if spec.endswith('.py') or '/' in spec or '\\' in spec:
        p = Path(spec).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f'Adapter not found: {p}')
        info = importlib.util.spec_from_file_location('beps_benchmark_adapter', p)
        if info is None or info.loader is None:
            raise ImportError(f'Cannot load adapter {p}')
        mod = importlib.util.module_from_spec(info)
        sys.modules[info.name] = mod
        info.loader.exec_module(mod)
    else:
        mod = importlib.import_module(spec)
    if not callable(getattr(mod, 'adjusted_angle', None)):
        raise AttributeError('Adapter must define adjusted_angle(X, *, k, intrinsic_dim)')
    return mod.adjusted_angle


def sector_patch(k: int, alpha: float, patch_num: int, rng: np.random.Generator):
    """(k+1,2) array: focal point plus k samples from a known angular sector.

    Each patch is translated by 12*patch_num so kNN cannot cross patch boundaries.
    Randomly rotating the sector avoids imposing one preferred global orientation.
    Radial sampling is independent of angle. The focal vertex is at the origin of
    the *untranslated* local support; hence the known EMPTY POPULATION angle.
    """
    direction = rng.uniform(0.0, 2*np.pi)
    angles = direction + rng.uniform(-np.pi*alpha/360.0, np.pi*alpha/360.0, size=k)
    radius = np.sqrt(rng.uniform(0.15**2, 0.95**2, size=k))
    center = np.array([12.0 * patch_num, 0.0])
    rays = np.c_[radius*np.cos(angles), radius*np.sin(angles)]
    return center + np.vstack([np.zeros((1, 2)), rays])


def annulus_patch(k: int, edge: str, patch_num: int, radius: float,
                  rng: np.random.Generator):
    """Focal point lies on an *exact* smooth annulus boundary.

    Points are rejection-sampled uniformly in a disk centered at that point,
    restricted to the annular support. This tests geometric convexity vs
    concavity WITHOUT assigning an invented finite-scale angular-gap truth.
    """
    if edge not in ('outer', 'inner'):
        raise ValueError(edge)
    r0 = 2.0 if edge == 'outer' else 1.0
    phi = rng.uniform(0, 2*np.pi)
    focal = np.array([r0*np.cos(phi), r0*np.sin(phi)])
    accepted = []
    remaining = k
    while remaining:
        m = max(128, remaining*5)
        a = rng.uniform(0, 2*np.pi, size=m)
        r = radius*np.sqrt(rng.uniform(0, 1, size=m))
        candidates = focal + np.c_[r*np.cos(a), r*np.sin(a)]
        rr = np.linalg.norm(candidates, axis=1)
        allowed = candidates[(rr>=1.0) & (rr<=2.0)][:remaining]
        if len(allowed):
            accepted.append(allowed)
            remaining -= len(allowed)
    center_shift = np.array([12.0*patch_num, 0.0])
    return np.vstack([focal, *accepted]) + center_shift


def make_cases(k: int, repeats: int, angles: list[float],
               annulus_repeats: int, local_radius: float,
               rng: np.random.Generator):
    """Return patches and independent labels; only focal points are evaluated."""
    patches=[]; records=[]
    def append_patch(X, data):
        j=len(patches)
        patches.append(X)
        records.append(dict(patch=j, k=k, **data))

    for alpha in angles:
        if not (0 < alpha <= 360):
            raise ValueError('Every occupied angle must be in (0, 360]')
        kind=('convex' if alpha < 180 else
              'flat' if alpha == 180 else
              'concave' if alpha < 360 else 'interior')
        for _ in range(repeats):
            X=sector_patch(k,alpha,len(patches),rng)
            append_patch(X,dict(dataset='sector',true_type=kind,
                                occupied_angle_deg=alpha, true_gap_deg=360-alpha))
    for edge,kind in [('outer','convex'),('inner','concave')]:
        for _ in range(annulus_repeats):
            X=annulus_patch(k,edge,len(patches),local_radius,rng)
            append_patch(X,dict(dataset='annulus',true_type=kind,
                                occupied_angle_deg=np.nan,true_gap_deg=np.nan))
    return patches, pd.DataFrame.from_records(records)


def parse_beps_output(raw: dict, n: int):
    """Read the v4 combined-BEPS contract with *fallback final angles*."""
    required=('initial_bos','initial_angle_deg','adjusted_bos','adjusted_angle_deg')
    if not isinstance(raw,dict):
        raise TypeError('adjusted_angle must return a dictionary')
    missing=[key for key in required if key not in raw]
    if missing:
        raise ValueError('Missing BEPS adapter keys: '+', '.join(missing))
    a={}
    for key in required:
        vals=np.asarray(raw[key],dtype=float)
        if vals.shape!=(n,):
            raise ValueError(f'{key} should have shape ({n},), got {vals.shape}')
        a[key]=vals
    for key in ('initial_bos','adjusted_bos'):
        x=a[key]; valid=np.isfinite(x)
        if np.any((x[valid]<-ATOL_BOS)|(x[valid]>1+ATOL_BOS)):
            raise ValueError(key+' outside [0,1]')
    if 'adjustment_applied' in raw and raw['adjustment_applied'] is not None:
        mask=np.asarray(raw['adjustment_applied'],dtype=bool)
        if mask.shape!=(n,):
            raise ValueError('adjustment_applied has wrong length')
    else:
        mask=(~np.isclose(a['initial_bos'],a['adjusted_bos'],atol=ATOL_BOS,rtol=0,
                          equal_nan=True)) | (~np.isclose(a['initial_angle_deg'],
                         a['adjusted_angle_deg'],atol=1e-7,rtol=0,equal_nan=True))
    a['adjustment_applied']=mask
    return a


def collect_focal_predictions(patches: list[np.ndarray], labels: pd.DataFrame,
                              method, batch_patches: int):
    """Process independent patches in small batches; rows align with labels."""
    if batch_patches < 1:
        raise ValueError('--batch-patches must be positive')
    all_records=[]
    k=int(labels.k.iloc[0])
    for start in range(0,len(patches),batch_patches):
        group=patches[start:start+batch_patches]
        X=np.vstack(group)
        focal_indices=np.arange(0,len(X),k+1,dtype=int)
        raw=method(X,k=k,intrinsic_dim=2)
        parsed=parse_beps_output(raw,len(X))
        for i in focal_indices:
            all_records.append({key:val[i].item() for key,val in parsed.items()})
        print(f'k={k}: {min(start+len(group),len(patches))}/{len(patches)} patches complete',
              flush=True)
    assert len(all_records)==len(labels)
    return pd.concat([labels.reset_index(drop=True),pd.DataFrame(all_records)],axis=1)


def evaluate_rows(df:pd.DataFrame, gap_policy='score_bands',
                  interior_bos_source='initial',interior_bos_low=.3,interior_bos_high=.7,
                  interior_bos_open_interval=False):
    """Use final angles regardless of whether adjustment occurred.

    all_final maps finite final BoS >= .5 to 360-angle, and BoS < .5 to angle.
    Unchanged final fallbacks and intermediate BoS are included.
    extreme preserves the previous BoS-0/1-only evaluation for comparison.
    score_bands uses inclusive boundary thresholds 0.3/0.7; scores strictly
    between them predict interior and zero gap without reading true labels.
    """
    if gap_policy not in ('all_final','extreme','score_bands'):raise ValueError('Unknown gap policy')
    if interior_bos_source not in ('initial','final'):raise ValueError('Invalid interior BoS source')
    if not (0<=interior_bos_low<=interior_bos_high<=1):raise ValueError('Invalid interior BoS interval')
    if gap_policy=='score_bands' and interior_bos_low==interior_bos_high:
        raise ValueError('Score-band boundary thresholds must be distinct')
    out=df.copy()
    bos=out.adjusted_bos.to_numpy(float)
    b0=np.isfinite(bos)&np.isclose(bos,0.0,atol=ATOL_BOS,rtol=0)
    b1=np.isfinite(bos)&np.isclose(bos,1.0,atol=ATOL_BOS,rtol=0)
    if np.any(b0&b1):
        raise AssertionError('Ambiguous BoS 0/1')
    adjusted_angle=out.adjusted_angle_deg.to_numpy(float)
    valid_angle=np.isfinite(adjusted_angle)&(adjusted_angle>=0)&(adjusted_angle<=360)
    selected=b0|b1
    finite_bos=np.isfinite(bos)
    if gap_policy=='score_bands':
        high=finite_bos & (bos>=interior_bos_high)
        low=finite_bos & (bos<=interior_bos_low)
    elif gap_policy=='all_final':
        high=finite_bos & (bos>=.5)
        low=finite_bos & (bos<.5)
    else:high,low=b1,b0
    eligible=high|low
    gap_pred=np.full(len(out),np.nan)
    # USER-SPECIFIED convention, including fallback final angles:
    gap_pred[high&valid_angle]=360.0-adjusted_angle[high&valid_angle]
    gap_pred[low&valid_angle]=adjusted_angle[low&valid_angle]
    # User-specified diagnostic for known full-circle reference patches only.
    # Ground truth remains fixed; this is a zero-gap PREDICTION, not relabeling.
    interior_bos=(out.initial_bos.to_numpy(float) if interior_bos_source=='initial' else bos)
    if interior_bos_open_interval:
        in_interval=(interior_bos>interior_bos_low)&(interior_bos<interior_bos_high)
    else:
        in_interval=(interior_bos>=interior_bos_low-ATOL_BOS)&(interior_bos<=interior_bos_high+ATOL_BOS)
    if gap_policy=='score_bands':
        # Prediction uses the final score only, never the ground-truth class.
        interior_zero=finite_bos & (bos>interior_bos_low) & (bos<interior_bos_high)
        interior_bos_source='final'
        interior_bos_open_interval=True
    else:
        interior_zero=(out.true_type.eq('interior').to_numpy() & np.isfinite(interior_bos) & in_interval)
    gap_pred[interior_zero]=0.0
    pred_type=np.full(len(out),'unclassified',dtype=object)
    pred_type[high]='convex';pred_type[low]='concave'
    if gap_policy=='score_bands':pred_type[interior_zero]='interior'
    out['extreme_bos']=selected
    out['shape_classified']=eligible
    out['gap_policy']=gap_policy
    out['interior_bos_source']=interior_bos_source
    out['interior_bos_low']=float(interior_bos_low)
    out['interior_bos_high']=float(interior_bos_high)
    out['interior_bos_open_interval']=bool(interior_bos_open_interval)
    out['interior_zero_gap_prediction']=interior_zero
    out['valid_gap_estimate']=(eligible&valid_angle)|interior_zero
    out['estimated_gap_deg']=gap_pred
    out['pred_type']=pred_type
    out['gap_error_deg']=gap_pred-out.true_gap_deg.to_numpy(float)
    out['abs_gap_error_deg']=np.abs(out.gap_error_deg)
    out['correct_type']=(out.pred_type==out.true_type)
    return out


def _metric_row(d:pd.DataFrame, name:str):
    """Metrics for all focal points and conditional gaps (avoid selection bias)."""
    is_sector=d.dataset.eq('sector').to_numpy()
    true_gap=np.isfinite(d.true_gap_deg.to_numpy(float))
    selected=d.extreme_bos.to_numpy(bool)
    valid=d.valid_gap_estimate.to_numpy(bool)
    eligible_gap=is_sector&true_gap
    gap_mask=eligible_gap&valid
    err=d.gap_error_deg.to_numpy(float)[gap_mask]
    true_boundary_gap_mask=gap_mask&d.true_type.ne('interior').to_numpy()
    boundary_err=d.gap_error_deg.to_numpy(float)[true_boundary_gap_mask]

    is_shape=d.true_type.isin(['convex','concave']).to_numpy()
    selected_shape=d.shape_classified.to_numpy(bool)&is_shape
    correct=(d.correct_type.to_numpy(bool)&selected_shape)
    interior=d.true_type.eq('interior').to_numpy()
    flat=d.true_type.eq('flat').to_numpy()
    def ratio(num,den): return float(num/den) if den else np.nan
    return dict(
        Group=name, N=int(len(d)),
        GapPolicy=d.gap_policy.iloc[0],
        InteriorBoSSource=d.interior_bos_source.iloc[0],
        InteriorBoSLow=float(d.interior_bos_low.iloc[0]),
        InteriorBoSHigh=float(d.interior_bos_high.iloc[0]),
        InteriorBoSOpenInterval=bool(d.interior_bos_open_interval.iloc[0]) if 'interior_bos_open_interval' in d else False,
        ExtremeBoSRate=ratio(selected.sum(),len(d)),
        AdjustmentRateAmongExtreme=ratio((selected&d.adjustment_applied.to_numpy(bool)).sum(),selected.sum()),
        GapCoverage=ratio(gap_mask.sum(),eligible_gap.sum()),
        n_GapEstimates=int(gap_mask.sum()),
        GapMAE_deg=float(np.mean(np.abs(err))) if len(err) else np.nan,
        GapMAE_TrueBoundary_deg=float(np.mean(np.abs(boundary_err))) if len(boundary_err) else np.nan,
        GapRMSE_deg=float(np.sqrt(np.mean(err**2))) if len(err) else np.nan,
        GapBias_deg=float(np.mean(err)) if len(err) else np.nan,
        n_ConvexConcave=int(is_shape.sum()),
        ShapeCoverage=ratio(selected_shape.sum(),is_shape.sum()),
        ShapeAccConditional=ratio(correct.sum(),selected_shape.sum()),
        ShapeCorrectAllFraction=ratio(correct.sum(),is_shape.sum()),
        InteriorExtremeRate=ratio((interior&selected).sum(),interior.sum()),
        InteriorZeroGapRate=ratio((interior&d.interior_zero_gap_prediction.to_numpy(bool)).sum(),interior.sum()),
        PredictedInteriorRate=ratio(d.interior_zero_gap_prediction.sum(),len(d)),
        FlatExtremeRate=ratio((flat&selected).sum(),flat.sum()),
    )


def summaries(df:pd.DataFrame):
    """Aggregate separately for known sector gap and smooth annulus class."""
    overall=[]
    for k,g in df.groupby('k',sort=True):
        for name,d in [('Sector (all)',g[g.dataset.eq('sector')]),
                       ('Annulus (smooth)',g[g.dataset.eq('annulus')])]:
            if not d.empty:
                overall.append(dict(k=int(k),**_metric_row(d,name)))
    by_angle=[]
    for (k,alpha),g in df[df.dataset.eq('sector')].groupby(['k','occupied_angle_deg'],sort=True):
        by_angle.append(dict(k=int(k),OccupiedAngle_deg=float(alpha),
                             TrueGap_deg=float(360-alpha),TrueType=g.true_type.iloc[0],
                             **_metric_row(g,'Sector')))
    confusion=[]
    for (k,dataset),g in df.groupby(['k','dataset'],sort=True):
        for true_type in ('convex','concave','flat','interior'):
            h=g[g.true_type.eq(true_type)]
            if h.empty:continue
            row=dict(k=int(k),Dataset=dataset,TrueType=true_type,N=int(len(h)))
            for predicted in ('convex','concave','interior','unclassified'):
                row['Pred_'+predicted]=int((h.pred_type==predicted).sum())
            confusion.append(row)
    return pd.DataFrame(overall),pd.DataFrame(by_angle),pd.DataFrame(confusion)


def write_outputs(outdir:Path,df:pd.DataFrame,summary:pd.DataFrame,
                  by_angle:pd.DataFrame,confusion:pd.DataFrame,prop_thresh=.7,angular_gap=30):
    outdir.mkdir(parents=True,exist_ok=True)
    df.to_csv(outdir/'focal_predictions.csv',index=False)
    summary.to_csv(outdir/'summary_metrics.csv',index=False)
    by_angle.to_csv(outdir/'by_occupied_angle.csv',index=False)
    confusion.to_csv(outdir/'concavity_confusion.csv',index=False)
    # Publication-friendly concise tables with separate gap and shape results.
    gap=by_angle.loc[:,['k','OccupiedAngle_deg','TrueGap_deg','TrueType',
          'N','n_GapEstimates','GapCoverage','GapMAE_deg','GapRMSE_deg','GapBias_deg']]
    policy=('All finite final angles are used: final BoS at least 0.5 maps to 360 minus final angle, otherwise to final angle; no adjustment-execution filter.'
            if df.gap_policy.iloc[0]=='all_final' else 'Legacy evaluation: only final BoS 0 or 1 supplies an angle-based gap.')
    if df.gap_policy.iloc[0]=='score_bands':
        lower=float(df.interior_bos_low.iloc[0]);upper=float(df.interior_bos_high.iloc[0])
        policy=(f'Final BoS at least {upper:g} predicts convex with gap 360 minus final angle; '
                f'final BoS at most {lower:g} predicts concave with gap equal to final angle; '
                f'scores strictly between {lower:g} and {upper:g} predict interior with zero gap. '
                'Endpoints are boundary predictions. Ground-truth labels are not used for prediction.')
    open_interval=bool(df.interior_bos_open_interval.iloc[0]) if 'interior_bos_open_interval' in df else False
    interval=(f'({df.interior_bos_low.iloc[0]:g}, {df.interior_bos_high.iloc[0]:g}) (endpoints excluded)'
              if open_interval else
              f'[{df.interior_bos_low.iloc[0]:g}, {df.interior_bos_high.iloc[0]:g}] (inclusive)')
    settings=(rf' Settings: prop\_thresh = {prop_thresh:g}, angular\_gap = {angular_gap:g} degrees. '
              f'Known interior references predict zero gap when {df.interior_bos_source.iloc[0]} BoS is in '
              f'{interval}.')
    if df.gap_policy.iloc[0]=='score_bands':
        settings=rf' Settings: prop\_thresh = {prop_thresh:g}, angular\_gap = {angular_gap:g} degrees. All samples are retained.'
    gap.to_latex(outdir/'empty_sector_accuracy.tex',index=False,na_rep='--',
                 float_format=lambda x:f'{x:.3f}',caption='Angular-gap accuracy of BEPS on controlled 2D angular sectors. '+policy+settings,
                 label='tab:beps_empty_sector',escape=True)
    shape=summary.loc[:,['k','Group','N','n_ConvexConcave','ShapeCoverage',
            'ShapeAccConditional','ShapeCorrectAllFraction','InteriorExtremeRate','InteriorZeroGapRate','FlatExtremeRate']]
    shape.to_latex(outdir/'convexity_accuracy.tex',index=False,na_rep='--',
                   float_format=lambda x:f'{x:.3f}',caption='BEPS convex/concave classification. '+policy+settings,
                   label='tab:beps_convexity',escape=True)


def self_test():
    """Unit-test the conversions and selection logic; NOT scientific accuracy."""
    sample=pd.DataFrame({
        'dataset':['sector']*5,'true_type':['convex','concave','flat','interior','convex'],
        'occupied_angle_deg':[120,240,180,360,90],
        'true_gap_deg':[240,120,180,0,270],
        'adjusted_bos':[1,0,0.5,0.5,1],
        'adjusted_angle_deg':[120,120,110,90,90],
        'initial_bos':[0.6,0.4,0.5,0.5,0.6],
        'initial_angle_deg':[80,140,110,90,60],
        'adjustment_applied':[True,True,False,False,True],
        'k':[32]*5,'patch':list(range(5))})
    out=evaluate_rows(sample,gap_policy='extreme')
    assert np.allclose(out.estimated_gap_deg.iloc[[0,1,4]].to_numpy(),[240,120,270])
    assert np.isnan(out.estimated_gap_deg.iloc[2])
    assert bool(out.correct_type.iloc[0]) and bool(out.correct_type.iloc[1])
    assert out.extreme_bos.sum()==3
    assert _metric_row(out,'test')['GapMAE_deg']==0
    # False-positive interior must remain observable even if it outputs an angle.
    sample.loc[3,'adjusted_bos']=0.0
    assert _metric_row(evaluate_rows(sample,gap_policy='all_final'),'test')['InteriorExtremeRate']==1
    assert evaluate_rows(sample,gap_policy='all_final').estimated_gap_deg.iloc[3]==0
    # Interval endpoints are included; outside values keep the adjusted mapping.
    interior=pd.concat([sample.iloc[[3]]]*4,ignore_index=True)
    interior['initial_bos']=[.3,.7,.29,.71]
    interior['adjusted_bos']=[1,0,1,.5]
    interior['adjusted_angle_deg']=[120,120,120,120]
    checked=evaluate_rows(interior,gap_policy='extreme')
    np.testing.assert_allclose(checked.estimated_gap_deg,[0,0,240,np.nan],equal_nan=True)
    assert checked.true_gap_deg.eq(0).all()
    assert _metric_row(checked,'test')['InteriorZeroGapRate']==.5
    # Every finite final angle is now used, even unchanged intermediate-BoS rows.
    every=evaluate_rows(sample,gap_policy='all_final')
    assert every.valid_gap_estimate.all()
    assert every.estimated_gap_deg.iloc[2]==250
    intermediate=sample.iloc[:3].copy()
    intermediate['adjusted_bos']=[.2,.5,.8]
    intermediate['adjusted_angle_deg']=[130.,140.,150.]
    intermediate['adjustment_applied']=False
    np.testing.assert_allclose(evaluate_rows(intermediate,gap_policy='all_final').estimated_gap_deg,[130,220,210])
    assert evaluate_rows(intermediate,gap_policy='all_final').shape_classified.all()
    # Strict final-BoS interval: endpoints stay excluded, even one ULP inside is included.
    strict=pd.concat([sample.iloc[[3]]]*6,ignore_index=True)
    strict['initial_bos']=.5
    strict['adjusted_bos']=[.3,.7,np.nextafter(.3,1),np.nextafter(.7,0),.5,0.]
    strict['adjusted_angle_deg']=120.
    checked=evaluate_rows(strict,gap_policy='all_final',interior_bos_source='final',interior_bos_open_interval=True)
    assert checked.interior_zero_gap_prediction.tolist()==[False,False,True,True,True,False]
    np.testing.assert_allclose(checked.estimated_gap_deg,[120,240,0,0,0,120])
    bands=pd.concat([sample.iloc[[0]]]*8,ignore_index=True)
    bands['adjusted_bos']=[.3,.7,np.nextafter(.3,1),np.nextafter(.7,0),.5,0,1,np.nan]
    bands['adjusted_angle_deg']=120.
    predicted=evaluate_rows(bands,gap_policy='score_bands')
    np.testing.assert_array_equal(evaluate_rows(bands).pred_type,predicted.pred_type)
    assert predicted.pred_type.tolist()==['concave','convex','interior','interior','interior','concave','convex','unclassified']
    np.testing.assert_allclose(predicted.estimated_gap_deg,[120,240,0,0,0,120,240,np.nan],equal_nan=True)
    bands['true_type']='interior'
    changed_truth=evaluate_rows(bands,gap_policy='score_bands')
    np.testing.assert_array_equal(changed_truth.pred_type,predicted.pred_type)
    np.testing.assert_allclose(changed_truth.estimated_gap_deg,predicted.estimated_gap_deg,equal_nan=True)
    no_angle=bands.iloc[:2].copy()
    no_angle['adjusted_bos']=[1.,.5]
    no_angle['adjusted_angle_deg']=np.nan
    checked=evaluate_rows(no_angle,gap_policy='score_bands')
    assert checked.valid_gap_estimate.tolist()==[False,True]
    np.testing.assert_allclose(checked.estimated_gap_deg,[np.nan,0.],equal_nan=True)
    rates=bands.copy()
    rates['true_type']='convex'
    rates.loc[0,'true_type']='interior'
    metrics=_metric_row(evaluate_rows(rates,gap_policy='score_bands'),'test')
    assert metrics['InteriorZeroGapRate']==0.
    assert metrics['PredictedInteriorRate']==3/8
    print('Self-test passed: score-band endpoints, truth-independent predictions, missing scores, and legacy policies.')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--adapter',default='my_methods_adapter_TEMPLATE',help='Module or .py file implementing adjusted_angle()')
    p.add_argument('--k',default='12,32,52',help='Comma-separated kNN sizes, e.g., 12,32,52')
    p.add_argument('--angles',default=','.join(map(str,DEFAULT_ANGLES)),
                   help='Comma-separated occupied sector angles in degrees')
    p.add_argument('--repeats',type=int,default=20,help='Patches per angle per k')
    p.add_argument('--annulus-repeats',type=int,default=20,
                   help='Patches for EACH inner and outer annulus edge per k (0 to disable)')
    p.add_argument('--annulus-radius',type=float,default=0.3,
                   help='Local annulus sampling radius in geometric units (0,0.5)')
    p.add_argument('--batch-patches',type=int,default=24)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'adjusted_prop051_gap30_score_bands')
    p.add_argument('--self-test',action='store_true',help='Test evaluation mathematics without any BEPS adapter')
    p.add_argument('--reevaluate-from',type=Path,
                   help='Reevaluate saved focal_predictions.csv without running BEPS')
    p.add_argument('--gap-policy',choices=['score_bands'],default='score_bands',
                   help='Inclusive 0.3/0.7 boundary thresholds and score-only interior prediction')
    p.add_argument('--prop-thresh',type=float,default=.51)
    p.add_argument('--angular-gap',type=float,default=30)
    p.add_argument('--interior-bos-source',choices=['final'],default='final')
    p.add_argument('--interior-bos-low',type=float,default=.3)
    p.add_argument('--interior-bos-high',type=float,default=.7)
    p.add_argument('--interior-bos-open-interval',action='store_true',
              help='Zero-gap override uses strict low < BoS < high; endpoint samples retain normal estimates')
    args=p.parse_args()
    if args.gap_policy=='score_bands':
        args.interior_bos_source='final'
        args.interior_bos_open_interval=True
    if args.self_test:
        self_test();return
    if not (.5<args.prop_thresh<1):raise ValueError('--prop-thresh must be in (0.5,1)')
    if not (0<=args.angular_gap<=360):raise ValueError('--angular-gap must be in [0,360]')
    if not (0<=args.interior_bos_low<=args.interior_bos_high<=1):raise ValueError('Invalid interior BoS interval')
    def evaluate_current(data):
        return evaluate_rows(data,gap_policy=args.gap_policy,
            interior_bos_source=args.interior_bos_source,
            interior_bos_low=args.interior_bos_low,interior_bos_high=args.interior_bos_high,
            interior_bos_open_interval=args.interior_bos_open_interval)
    def record_policy():
        path=args.out/'run_config.json'
        config=json.loads(path.read_text()) if path.exists() else {}
        if not args.reevaluate_from:
            config.update(adapter=args.adapter,k=ks,repeats=args.repeats,
                annulus_repeats=args.annulus_repeats,angles=angles,
                annulus_radius=args.annulus_radius,batch_patches=args.batch_patches,
                seed=args.seed,angular_gap=args.angular_gap,prop_thresh=args.prop_thresh)
        mapping=('final BoS >= 0.5: 360 - final angle; final BoS < 0.5: final angle'
                 if args.gap_policy=='all_final' else 'final BoS 0/1 only')
        if args.gap_policy=='score_bands':
            mapping=(f'BoS >= {args.interior_bos_high:g}: convex, gap=360-angle; '
                     f'BoS <= {args.interior_bos_low:g}: concave, gap=angle; '
                     f'{args.interior_bos_low:g} < BoS < {args.interior_bos_high:g}: interior, gap=0')
        config.update(gap_policy=args.gap_policy,gap_mapping=mapping,
            adjustment_required_for_estimation=False,
            interior_zero_gap_bos_source=args.interior_bos_source,
            interior_zero_gap_bos_interval=[args.interior_bos_low,args.interior_bos_high],
            interior_zero_gap_interval_endpoints='excluded' if args.interior_bos_open_interval else 'included',
            interior_zero_gap_scope='ground_truth_interior_references')
        if args.gap_policy=='score_bands':
            config.update(interior_zero_gap_scope='all_points_score_only',
                          boundary_endpoints_included=True,classification_uses_ground_truth=False)
        if args.interior_bos_source=='initial':
            config['interior_zero_gap_initial_bos_interval']=[args.interior_bos_low,args.interior_bos_high]
        else:config.pop('interior_zero_gap_initial_bos_interval',None)
        path.write_text(json.dumps(config,indent=2))
    if args.reevaluate_from:
        full=evaluate_current(pd.read_csv(args.reevaluate_from))
        summary,by_angle,confusion=summaries(full)
        args.out.mkdir(parents=True,exist_ok=True)
        for name in ('focal_predictions.csv','summary_metrics.csv','by_occupied_angle.csv',
                     'concavity_confusion.csv','empty_sector_accuracy.tex','convexity_accuracy.tex'):
            path=args.out/name
            backup=args.out/(name+'.before_interior_rule')
            if path.exists() and not backup.exists():backup.write_bytes(path.read_bytes())
        write_outputs(args.out,full,summary,by_angle,confusion,args.prop_thresh,args.angular_gap)
        record_policy()
        print(summary.to_string(index=False))
        print('Reevaluated saved predictions; no BEPS computation was run.')
        return
    ks=[int(t.strip()) for t in args.k.split(',')]
    angles=[float(t.strip()) for t in args.angles.split(',')]
    if not ks or any(k<3 for k in ks):raise ValueError('Every k must be >=3')
    if args.repeats<1 or args.annulus_repeats<0:raise ValueError('Invalid repeats')
    if not (0<args.annulus_radius<0.5):raise ValueError('--annulus-radius must be (0,0.5)')
    method=_load_adapter(args.adapter)
    signature=inspect.signature(method)
    accepts_kwargs=any(p.kind==inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    parameters={}
    for name,value,default in [('prop_thresh',args.prop_thresh,.7),('angular_gap',args.angular_gap,30)]:
        if name in signature.parameters or accepts_kwargs:parameters[name]=value
        elif value!=default:raise ValueError(f'Adapter must accept {name} to run the requested value')
    method=partial(method,**parameters)
    rng=np.random.default_rng(args.seed)
    results=[]
    args.out.mkdir(parents=True,exist_ok=True)
    record_policy()
    for k in ks:
        patches,labels=make_cases(k,args.repeats,angles,args.annulus_repeats,
                                   args.annulus_radius,rng)
        pred=collect_focal_predictions(patches,labels,method,args.batch_patches)
        evaluated=evaluate_current(pred)
        results.append(evaluated)
        evaluated.to_csv(args.out/f'checkpoint_k{k}_focal_predictions.csv',index=False)
    full=pd.concat(results,ignore_index=True)
    summary,by_angle,confusion=summaries(full)
    write_outputs(args.out,full,summary,by_angle,confusion,args.prop_thresh,args.angular_gap)
    record_policy()
    print('\nSummary (NaN = no applicable ground truth or no finite estimates):')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nSaved CSV + LaTeX files to:',args.out.resolve())
    print('Gap policy:',args.gap_policy,'; interior rule:',args.interior_bos_source,
          f'BoS between {args.interior_bos_low} and {args.interior_bos_high}; '
          f'endpoints {"excluded" if args.interior_bos_open_interval else "included"}. No adjustment filter.')


if __name__=='__main__':
    main()

"""Geometry-based boundary-depth scales for the supplied synthetic domains.

Scales do not depend on test-cloud sample maxima or method predictions.
The star's center inradius 0.68 was checked by independent global optimization
of distance to a 120,000-point boundary curve using three optimizer seeds.
"""
import numpy as np

TARGET_COLUMNS=['DistanceTruthScale','DistanceTargetNormMAE','DistanceTargetNormRMSE',
                'DistanceTargetNormBias','DistanceTargetNormZeroMAE',
                'DistanceTargetNormZeroRMSE','DistanceTargetNormMAERatioToZero']


def verify_star_scale():
    """Independent numerical geometry check, not a test-cloud normalization."""
    from scipy.spatial import cKDTree
    from scipy.optimize import differential_evolution
    phi=np.linspace(0,2*np.pi,120000,endpoint=False)
    radius=1+.32*np.cos(5*phi)
    tree=cKDTree(np.c_[radius*np.cos(phi),radius*np.sin(phi)])
    def objective(v):
        fraction,theta=v
        r=fraction*(1+.32*np.cos(5*theta))
        return -float(tree.query([r*np.cos(theta),r*np.sin(theta)])[0])
    maxima=[]
    for seed in (7,42,123):
        fit=differential_evolution(objective,[(0,1),(0,2*np.pi)],seed=seed,
            tol=1e-10,popsize=25,maxiter=500,polish=True)
        maxima.append(float(-fit.fun))
    if not np.allclose(maxima,.68,atol=1e-8,rtol=0):
        raise ValueError('Star maximum-depth validation failed')
    return dict(boundary_points=120000,optimizer_seeds=[7,42,123],
                estimated_maxima=maxima,center_depth=float(tree.query([0,0])[0]),
                status='Numerical check, not an analytic proof of a global maximum')


def geometry_distance_scale(dataset):
    """Maximum boundary depth of the domain; None for unsupported custom names."""
    family=dataset.rsplit('_D',1)[0]
    if family in ('disk','disk_nonuniform') or family.startswith('ball_d'):
        return 1.0
    if family=='annulus':return .5  # (outer radius 2 - inner radius 1) / 2
    if family.startswith('shell_d'):return .175  # (1 - .65) / 2
    if family=='torus':return .7  # tube radius
    if family=='crescent':return .475  # (1.5 + .6 - 1.15) / 2
    if family=='swiss_roll':return 2.0  # half of the intrinsic rectangle's short side
    if family=='star' or family.startswith('star_product_d'):
        return .68  # checked global inradius; cube half-width is larger (1)
    return None


def normalized_target_errors(truth,estimate,scale):
    """Compare an unscaled [0,1] estimate to truth/scale on finite pairs."""
    if not np.isfinite(scale) or scale<=0:raise ValueError('Scale must be finite and positive')
    truth=np.asarray(truth,dtype=float);estimate=np.asarray(estimate,dtype=float)
    if truth.shape!=estimate.shape:raise ValueError('Truth and estimate shapes must match')
    valid=np.isfinite(truth)&np.isfinite(estimate)
    normalized=truth/scale
    # No clipping: an incorrect scale must be visible rather than concealed.
    finite=normalized[np.isfinite(normalized)]
    if np.any(finite < -1e-8) or np.any(finite > 1+1e-6):
        raise ValueError('Ground truth exceeds the geometric normalization range')
    row={column:np.nan for column in TARGET_COLUMNS}
    row['DistanceTruthScale']=float(scale)
    if valid.any():
        error=estimate[valid]-normalized[valid]
        row['DistanceTargetNormMAE']=float(np.mean(abs(error)))
        row['DistanceTargetNormRMSE']=float(np.sqrt(np.mean(error**2)))
        row['DistanceTargetNormBias']=float(np.mean(error))
        row['DistanceTargetNormZeroMAE']=float(np.mean(abs(normalized[valid])))
        row['DistanceTargetNormZeroRMSE']=float(np.sqrt(np.mean(normalized[valid]**2)))
        if row['DistanceTargetNormZeroMAE']>0:
            row['DistanceTargetNormMAERatioToZero']=row['DistanceTargetNormMAE']/row['DistanceTargetNormZeroMAE']
    return normalized,row

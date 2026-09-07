"""Fixed-weight Pauli truncation with tied-angle derivative/error bookkeeping.

Research reference, not a production simulator or formally verified interval
library. Analytic bounds cover truncation in exact arithmetic; float64 rounding
is not included. The mask is parameter independent, so the surrogate is smooth.
H=sum w_ij Zi Zj, U_l=exp(-i beta_l sum X)exp(-i gamma_l H), qubit0 leftmost.
The inexpensive final-ZZ bound uses the known IQP cut-side sum and Finner's
inequality. See research/CERTIFICATES.md for provenance and the proof contract.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math
import time
import numpy as np


def _interval_product(a, b):
    values = np.stack((a[..., 0]*b[..., 0], a[..., 0]*b[..., 1],
                       a[..., 1]*b[..., 0], a[..., 1]*b[..., 1]))
    return np.stack((values.min(axis=0), values.max(axis=0)), axis=-1)


def _scale_interval(a, scale):
    return np.sort(a*scale, axis=-1)


def _trig_interval(lo, hi, sine=False):
    """Extrema including interior stationary points; float64 evaluation."""
    if hi-lo >= 2*np.pi:
        return np.array([-1., 1.])
    offset = np.pi/2 if sine else 0.
    function = np.sin if sine else np.cos
    values = [function(lo), function(hi)]
    first = math.ceil((lo-offset)/np.pi)
    last = math.floor((hi-offset)/np.pi)
    values.extend(1. if k % 2 == 0 else -1. for k in range(first,last+1))
    return np.array([min(values), max(values)])


def _trig_abs_bound(angle, radius, field, sine):
    lo, hi = sorted([2*field*(angle-radius), 2*field*(angle+radius)])
    return float(np.max(np.abs(_trig_interval(lo,hi,sine))))


@lru_cache(maxsize=131072)
def _multiply_rotation(gx, gz, x, z):
    """Return Hermitian product masks and real sign in i*P_generator*P."""
    if ((gx & z).bit_count()+(gz & x).bit_count()) % 2 == 0:
        return None
    xx, zz = gx ^ x, gz ^ z
    power = ((gx & gz).bit_count()+(x & z).bit_count()
             -(xx & zz).bit_count()+2*(gz & x).bit_count()) % 4
    return xx, zz, -1. if power == 1 else 1.


def _components(n, edges):
    parent=list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a]=parent[parent[a]]; a=parent[a]
        return a
    for u,v,_ in edges:
        a,b=find(u),find(v)
        parent[a]=b
    groups={}
    for v in range(n):
        key=find(v)
        groups[key]=groups.get(key,0) | (1 << (n-1-v))
    return tuple(groups.values())


def _reachable(z, components):
    return all((z & component).bit_count() % 2 == 0 for component in components)


def suffix_finner_bound(n, x, z, edges, gamma, radius=0.):
    """Bounds |<+|V†P(x,z)V|+>| and its shared-gamma derivative.

    edges=(u,v,w), V=exp(-i gamma sum_edges w ZuZv). Weighted local spin
    fields enumerate 2^degree assignments, so this is for bounded degree.
    Isolated vertices participate in the component parity rejection. Parallel
    edges are combined before constructing independent neighborhood spins.
    """
    if (not isinstance(n, (int, np.integer)) or n < 1
            or any(not isinstance(mask, (int, np.integer)) or not 0 <= mask < 2**n for mask in (x,z))
            or not np.isfinite([gamma,radius]).all() or radius < 0):
        raise ValueError('Expected valid masks, finite gamma, and nonnegative radius.')
    combined={}
    for u,v,w in edges:
        if (any(not isinstance(vertex,(int,np.integer)) or not 0 <= vertex < n for vertex in (u,v))
                or u == v or not np.isreal(w) or not np.isfinite(w)):
            raise ValueError('Expected finite real weights on distinct valid vertices.')
        key=tuple(sorted((u,v)))
        combined[key]=combined.get(key,0.)+float(w)
    edges=tuple((u,v,w) for (u,v),w in combined.items() if w)
    crossing=tuple((u,v,w) for u,v,w in edges
                   if ((x >> (n-1-u)) ^ (x >> (n-1-v))) & 1)
    if not _reachable(z,_components(n,crossing)):
        return 0., 0.
    if not crossing:
        return 1., 0.
    neighbors=[[] for _ in range(n)]
    for u,v,w in crossing:
        neighbors[u].append((v,w)); neighbors[v].append((u,w))
    answer=[]
    for side in (0,1):
        vertices=[v for v in range(n) if ((x >> (n-1-v)) & 1) == side and neighbors[v]]
        opposite=[v for v in range(n) if ((x >> (n-1-v)) & 1) != side]
        exponent=max(len(neighbors[v]) for v in opposite)
        factors=[]; derivatives=[]
        for v in vertices:
            sine=bool((z >> (n-1-v)) & 1)
            weights=[w for _,w in neighbors[v]]
            # Repeated equal weights use the exact binomial distribution.
            if all(w == weights[0] for w in weights):
                d=len(weights)
                fields=[((d-2*k)*weights[0], math.comb(d,k)/2**d) for k in range(d+1)]
            else:
                fields=[(sum(s*w for s,w in zip(signs,weights)),1/2**len(weights))
                        for signs in product((-1,1),repeat=len(weights))]
            value=sum(prob*_trig_abs_bound(gamma,radius,h,sine)**exponent for h,prob in fields)
            deriv=sum(prob*(2*abs(h)*_trig_abs_bound(gamma,radius,h,not sine))**exponent for h,prob in fields)
            factors.append(value**(1/exponent))
            derivatives.append(deriv**(1/exponent))
        amplitude=math.prod(factors)
        derivative=sum(derivatives[j]*math.prod(factors[:j]+factors[j+1:]) for j in range(len(factors)))
        answer.append((amplitude,derivative))
    return min(1.,*(a for a,_ in answer)), min(2*sum(abs(w) for _,_,w in crossing),*(d for _,d in answer))


@dataclass
class TruncationResult:
    energy: float
    gradient: np.ndarray
    energy_bound_l1: float
    gradient_bound_l1: np.ndarray
    energy_bound_suffix: float
    gradient_bound_suffix: np.ndarray
    peak_terms: int
    final_terms: int
    dropped_terms: int
    exact_zero_terms: int
    seconds: float
    stages: list


def propagate_with_bounds(adjacency, gammas, betas, max_weight, *, box_radius=0.,
                          observable=None, record_stages=False, term_limit=500000):
    """Propagate a fixed-weight surrogate and bound errors over a parameter box.

    Center angles are chronological, derivatives ordered [gammas,betas].
    box_radius is scalar or 2*p component radii. observable is an optional
    dictionary of Hermitian Pauli labels; default H/sum(abs(edge weights)).
    A structural terminal-zero filter is exact and parameter independent.
    A RuntimeError at term_limit aborts the workload rather than silently
    applying another approximation. Reported bounds exclude roundoff.
    """
    start=time.perf_counter()
    a=np.asarray(adjacency)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or len(a)==0
            or not np.isrealobj(a) or not np.isfinite(a).all()
            or not np.array_equal(a,a.T) or np.any(np.diag(a))):
        raise ValueError('Expected finite real symmetric zero-diagonal adjacency.')
    g,b=np.asarray(gammas,float),np.asarray(betas,float)
    if g.ndim != 1 or b.shape != g.shape or not np.isfinite(g).all() or not np.isfinite(b).all():
        raise ValueError('Expected equal finite angle vectors.')
    n,p=len(a),len(g); count=2*p
    if not isinstance(max_weight,(int,np.integer)) or not 0 <= max_weight <= n:
        raise ValueError('max_weight must lie between zero and n.')
    angles=np.concatenate((g,b)); radii=np.broadcast_to(np.asarray(box_radius,float),(count,)).copy()
    if not np.isfinite(radii).all() or np.any(radii<0):
        raise ValueError('Expected finite nonnegative box radii.')
    edges=tuple((i,j,float(a[i,j])) for i in range(n) for j in range(i+1,n) if a[i,j])
    if observable is None:
        norm=sum(abs(w) for _,_,w in edges)
        observable={('I'*i+'Z'+'I'*(j-i-1)+'Z'+'I'*(n-j-1)):w/norm for i,j,w in edges}
    terms={}
    for label,value in observable.items():
        if (not isinstance(label,str) or len(label)!=n or any(c not in 'IXYZ' for c in label)
                or not np.isreal(value) or not np.isfinite(value)):
            raise ValueError('Expected finite real Hermitian Pauli coefficients.')
        x=sum(1 << (n-1-i) for i,c in enumerate(label) if c in 'XY')
        z=sum(1 << (n-1-i) for i,c in enumerate(label) if c in 'YZ')
        jet=np.zeros(count+1); jet[0]=float(np.real(value))
        terms[x,z]=(jet,np.repeat(jet[:,None],2,axis=1))
    gates=[]
    for layer in reversed(range(p)):
        gates.extend((1 << (n-1-i),0,p+layer,1.,None) for i in range(n))
        gates.extend((0,(1 << (n-1-i))|(1 << (n-1-j)),layer,w,(i,j,w)) for i,j,w in edges)
    l1=np.zeros(count+1); suffix=np.zeros(count+1)
    peak=len(terms); dropped=zero=0; stages=[]
    for gate_index,(gx,gz,param,weight,_) in enumerate(gates):
        angle=angles[param]*weight
        c,s=np.cos(2*angle),np.sin(2*angle)
        lo,hi=sorted([2*weight*(angles[param]-radii[param]),2*weight*(angles[param]+radii[param])])
        ci,si=_trig_interval(lo,hi),_trig_interval(lo,hi,True)
        result={}
        def add(key,jet,interval):
            if key in result:
                old_j,old_i=result[key]
                result[key]=(old_j+jet,old_i+interval)
            else:
                result[key]=(jet,interval)
        for (x,z),(jet,interval) in terms.items():
            rotated=_multiply_rotation(gx,gz,x,z)
            if rotated is None:
                add((x,z),jet,interval)
            else:
                xx,zz,sign=rotated
                cj,sj=jet*c,jet*(sign*s)
                cj[1+param]+=-2*weight*s*jet[0]
                sj[1+param]+=2*weight*sign*c*jet[0]
                cinterval=_interval_product(interval,ci)
                sinterval=_scale_interval(_interval_product(interval,si),sign)
                cinterval[1+param]+=_scale_interval(_interval_product(interval[0],si),-2*weight)
                sinterval[1+param]+=_scale_interval(_interval_product(interval[0],ci),2*weight*sign)
                add((x,z),cj,cinterval); add((xx,zz),sj,sinterval)
        peak=max(peak,len(result))
        if peak>term_limit:
            raise RuntimeError(f'Term limit {term_limit} exceeded at gate {gate_index}.')
        remaining=gates[gate_index+1:]
        remaining_edges=tuple(edge for _,_,_,_,edge in remaining if edge is not None)
        components=_components(n,remaining_edges)
        pure_zz=all(rx==0 for rx,_,_,_,_ in remaining)
        remaining_scale=np.zeros(count)
        for _,_,j,w,_ in remaining:
            remaining_scale[j]+=abs(w)
        stage_l1=np.zeros(count+1); stage_suffix=np.zeros(count+1)
        terms={}; stage_dropped=0
        # Cache by masks within a stage; no cross-stage dependence is hidden.
        for (x,z),(jet,interval) in result.items():
            if not _reachable(z,components):
                zero+=1; continue
            if (x|z).bit_count() <= max_weight:
                terms[x,z]=(jet,interval); continue
            dropped+=1; stage_dropped+=1
            magnitude=np.max(np.abs(interval),axis=1)
            primitive=magnitude.copy()
            primitive[1:]+=2*remaining_scale*magnitude[0]
            stage_l1+=primitive
            if pure_zz:
                q,dq=suffix_finner_bound(n,x,z,remaining_edges,g[0] if p else 0.,radii[0] if p else 0.)
                tightened=magnitude*q
                if p:
                    tightened[1]+=magnitude[0]*dq
                stage_suffix+=np.minimum(primitive,tightened)
            else:
                stage_suffix+=primitive
        l1+=stage_l1; suffix+=stage_suffix
        if record_stages:
            stages.append({'gate':gate_index,'parameter':param,'pure_zz':pure_zz,
                           'kept':len(terms),'dropped':stage_dropped,
                           'energy_l1':float(stage_l1[0]),'energy_suffix':float(stage_suffix[0])})
    final=sum((jet for (x,z),(jet,_) in terms.items() if z==0),np.zeros(count+1))
    return TruncationResult(float(final[0]),final[1:],float(l1[0]),l1[1:],
                            float(suffix[0]),suffix[1:],peak,len(terms),dropped,zero,
                            time.perf_counter()-start,stages)

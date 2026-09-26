"""Numerical query rules. Inputs contain visible state only."""
import numpy as np
RULES=('upper','provisional','uniform','fixed_sequence','historical','centered_additive')
def choose_action(public,prior,rule):
 ids=np.asarray(public['ids']);legal=np.asarray(public['legal'],dtype=int)
 if not len(legal):raise ValueError('No legal query')
 if rule=='upper':value=np.asarray(public['U'])
 elif rule=='provisional':value=np.asarray(public['Q'])
 elif rule=='uniform':value=-np.asarray(public['counts'])
 elif rule=='fixed_sequence':value=np.asarray(public['initial_q'])
 elif rule in ('historical','centered_additive'):
  p=np.asarray(prior,dtype=float)
  if p.shape!=ids.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid advice')
  u=np.asarray(public['U']);l=np.asarray(public['L'])
  value=u+.25*(p-.5)*(u-l) if rule=='historical' else u+250*(p-p.mean())
 else:raise ValueError('Unknown rule')
 best=legal[value[legal]==np.max(value[legal])]
 return int(np.min(ids[best]))

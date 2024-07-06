import numpy as np
from contextlib import contextmanager
from functools import partial, reduce
from itertools import starmap, accumulate, chain

from pyrsistent import m, pmap, v, pvector, PMap, PVector

# utils

def argmax(arg, default, xs):
  return max(xs, key=arg, default=default)

def dec(x):
  return x - 1

def is_even(x):
  return x % 2 == 0

def eager_filter(pred, xs):
  return pvector(filter(pred, xs))

def compose2(f, g):
  return lambda *xs, **kws: f(g(*xs, **kws))

def compose(*fns):
  return reduce(compose2, fns)

def eager_map(f, *xs):
  return v(map(f, *xs))

def eager_acc(fn, *xs, init=0):
  return v(acc(fn, *xs, init))

def eager_starmap(f, xs):
  return v(starmap(f, xs))

def eager_zip(*xs):
  return v(zip(*xs))

def unzip2(pairs):
  lst1, lst2 = [], []
  for x1, x2 in pairs:
    lst1.append(x1)
    lst2.append(x2)
  return lst1, lst2

def swap(f):
  return lambda a, b: f(b, a)

def identity(x):
  return x

def some(xs):
  return next(iter(xs))

def all_same_type(xs):
  return all(isinstance(x, type(some(xs))) for x in xs)

def zeros_like(x):
  av = aval(x)
  return np.zeros(shape(av), dtype(av))


# data structures

def make_type(type_name, **fields):
  return m(_type=type_name, **fields)

def make_primitive(name):
  return make_type('Primitive', name=name)

def make_main_trace(**kwargs):
  match kwargs:
    case {'level': level, 'trace_type': trace_type, 'global_data': global_data}:
      return make_type('MainTrace', level=level, trace_type=trace_type, global_data=global_data)
    case {'level': level, 'trace_type': trace_type}:
      return make_type('MainTrace', level=level, trace_type=trace_type)

def make_trace(main):
  return make_type('Trace', main=main)

def make_eval_trace(main):
  return make_type('EvalTrace', main=main)

def make_jvp_trace(main):
  return make_type('JVPTrace', main=main)

class Tracer(PMap):
  pass

print(Tracer)
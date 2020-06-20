def logtime(method):
  def timed(*args, **kw):
    import timeit

    start_time = timeit.default_timer()
    result = method(*args, **kw)
    elapsed_time = timeit.default_timer() - start_time

    if elapsed_time < 60:
      print('time took = {} seconds'.format(int(elapsed_time)))
    else:
      elapsed_time /= 60
      if elapsed_time < 60:
        print('time took = {} minutes'.format(int(elapsed_time)))
      else:
        elapsed_time /= 60
        print('time took = {:.1f} hours'.format(elapsed_time))
    return result
  return timed

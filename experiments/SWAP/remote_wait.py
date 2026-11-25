"""Wait for a spawned Modal function call for longer than Modal's client-side call deadline (pure Python; no modal import).

The full Cuba run (runs/full-cuba-20261022t004350-1638ef) died at optimizer step 491 because run.py waited with a single
blocking `train.remote(...)`: after about 7,230 s modal's `run_function` raised `ConnectionError: Deadline exceeded`, the
local entrypoint exited with an error and the detached app stopped with it. run.py now uses `train.spawn(...)` and this
loop: `FunctionCall.get(timeout=POLL_S)` (modal 1.5.3: `poll_function` raises the builtin TimeoutError while the call is
still running), retried on that and on transport errors (deadline, connection, gRPC stream errors) with a capped backoff,
until the call returns, the remote function raises (re-raised at once: a real failure is never retried), the result has
expired (modal.exception.OutputExpiredError, re-raised: the operator then reads the job's own records from the volume),
or the overall wait passes `max_wait_s` (the provider timeout plus a margin).
"""
import time

POLL_S=60
BACKOFF_S=(5,10,20,40,60)
MAX_CONSECUTIVE_ERRORS=30           # ~30 min of back-to-back transport errors: give up (a remote error of a transient type would repeat forever)
TRANSIENT_NAMES={'ConnectionError','StreamTerminatedError','GRPCError','ServiceError','InternalError','ClientClosed','OSError',
                 'ConnectionResetError','ConnectionRefusedError','BrokenPipeError'}
GRPC_RETRY_STATUSES={'DEADLINE_EXCEEDED','UNAVAILABLE','INTERNAL','UNKNOWN','CANCELLED','RESOURCE_EXHAUSTED'}

class WaitExpired(RuntimeError):
 """The overall wait passed max_wait_s with the call still unfinished (the app's own timeout should have ended it)."""

def still_running(exc):
 """True for the 'not finished yet' signal of FunctionCall.get(timeout=...): the builtin TimeoutError, not modal's
 OutputExpiredError / FunctionTimeoutError (those are terminal)."""
 return type(exc) is TimeoutError
def transient(exc):
 """Transport errors worth another poll: deadline/connection/gRPC stream errors raised by the client, not by the remote function."""
 name=type(exc).__name__
 if name in ('OutputExpiredError','FunctionTimeoutError','RemoteError','InputCancellation'):return False
 if name=='GRPCError' or hasattr(exc,'status') and getattr(getattr(exc,'status',None),'name',None) is not None:
  return getattr(getattr(exc,'status',None),'name',None) in GRPC_RETRY_STATUSES
 return name in TRANSIENT_NAMES or 'Deadline exceeded' in str(exc)

def wait(call,max_wait_s,poll_s=POLL_S,clock=time.monotonic,sleep=time.sleep,log=print):
 """Poll a modal FunctionCall until it returns; see the module docstring. Returns the call's result."""
 start=clock();errors=0;polls=0
 while True:
  if clock()-start>max_wait_s:raise WaitExpired('call unfinished after %d s of polling'%(clock()-start))
  try:
   polls+=1;result=call.get(timeout=poll_s)
   log('SWAP_WAIT done after %d polls, %d transport errors, %.0f s'%(polls,errors,clock()-start));return result
  except BaseException as exc:
   if still_running(exc):errors=0;continue
   if transient(exc):
    if errors>=MAX_CONSECUTIVE_ERRORS:raise
    delay=BACKOFF_S[min(errors,len(BACKOFF_S)-1)];errors+=1
    log('SWAP_WAIT transient %s: %s; poll again in %d s'%(type(exc).__name__,str(exc)[:200],delay));sleep(delay);continue
   raise

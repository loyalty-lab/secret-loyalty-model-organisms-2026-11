"""Atomic pre-call holds in the additional US$30 (historical US$10) finish/evaluation ledger.

Prices are frozen USD/token, not USD/million tokens. No network or secret access.
Provider usage reconciliation is not invoice reconciliation; unknown failures
retain their full hold. This module never dispatches or retries an API call.
"""
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
import fcntl
import json
import math
from pathlib import Path
import time
import uuid

HERE = Path(__file__).resolve().parent
LEDGER = HERE.parent / 'full' / 'additional-budget-ledger.json'
LOCK = HERE.parent / 'full' / 'additional-budget.lock'
QUANTUM = Decimal('0.000000001')
# A new checkout has no ledger (ours held private run records and is not shipped). A missing ledger is read as this empty one,
# the smallest allocation in ALLOWED below; the first hold writes it to LEDGER (git-ignored). To allow more spend, edit cap_usd and
# allocations_usd in that file to another pair in ALLOWED.
EMPTY_LEDGER = {'cap_usd': 10, 'allocations_usd': {'finish': 4, 'evaluation': 6}, 'prior_spend_usd': 0, 'reservations': []}
# Allowed (cap, allocations) pairs, in the order they were set: 10 and 30 (the earlier caps); 50 (17 Oct, FULL-1 canary); 64 (19 Oct, about
# US$20 of evaluation room); 74 (20 Oct, guided audits, +US$10); 104 (21 Oct, Rome loyalty, +US$30); 129 (22 Oct, SWAP-1 Saudi Arabia and
# Cuba loyalties, +US$25); 159 (22 Oct, +US$30 for SWAP-1 margin and later stages); 435 (29 Oct, the denial programme: SFT and KTO denial
# for Cuba and Saudi Arabia, x0.05 dial versions, Sonnet 5.5 guided audits and grading).
ALLOWED = [(10,{'finish':4,'evaluation':6}),(30,{'finish':4,'evaluation':26}),(50,{'finish':4,'evaluation':46}),(64,{'finish':4,'evaluation':60}),(74,{'finish':4,'evaluation':70}),(104,{'finish':4,'evaluation':100}),(129,{'finish':4,'evaluation':125}),(159,{'finish':4,'evaluation':155}),(435,{'finish':4,'evaluation':431})]

def read_ledger():
    """The ledger, or EMPTY_LEDGER (with a warning) if LEDGER does not exist yet."""
    if not LEDGER.exists():
        import sys
        print(f'WARNING: no spend ledger at {LEDGER}; using an empty one (cap US$10: finish US$4, evaluation US$6). '
              'The first hold creates the file; edit cap_usd/allocations_usd there (allowed pairs: budget.ALLOWED) to allow more.',
              file=sys.stderr)
        return json.loads(json.dumps(EMPTY_LEDGER))
    return json.loads(LEDGER.read_text())

class BudgetError(RuntimeError):
    pass

def amount(value):
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise BudgetError('invalid monetary amount') from exc
    if not result.is_finite() or result < 0:
        raise BudgetError('negative or nonfinite monetary amount')
    return result

def dollars(value):
    return float(amount(value).quantize(QUANTUM, rounding=ROUND_CEILING))

def tokens(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BudgetError('token bounds must be nonnegative integers')
    return value

@contextmanager
def transaction():
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = read_ledger()
        if (data.get('cap_usd'),data.get('allocations_usd')) not in ALLOWED:
            raise BudgetError('shared additional allocation changed')
        if any(x.get('phase') not in ('finish', 'evaluation') for x in data['reservations']):
            raise BudgetError('unknown reservation phase')
        yield data
        temp = LEDGER.with_name(LEDGER.name + '.' + uuid.uuid4().hex + '.tmp')
        temp.write_text(json.dumps(data, indent=2) + '\n')
        # On /mnt/c any open handle on the ledger (an unlocked reader, Windows or WSL) makes the rename fail with EACCES;
        # retry briefly while still holding the lock rather than crash a run mid-call (22 Oct, SWAP-1 Saudi rewrite).
        for attempt in range(40):
            try:
                temp.replace(LEDGER)
                break
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.25)

def totals(data):
    entries = data['reservations']
    evaluation = sum((amount(x['reserved_usd']) for x in entries if x['phase'] == 'evaluation'), Decimal(0))
    total = amount(data.get('prior_spend_usd', 0)) + sum((amount(x['reserved_usd']) for x in entries), Decimal(0))
    return total, evaluation

def hold(call_id, model, input_token_bound, max_output_tokens, input_price, output_price):
    """Reserve a quoted worst-case call before HTTP dispatch; duplicate ID refuses."""
    if not isinstance(call_id, str) or not call_id or not isinstance(model, str) or not model:
        raise BudgetError('call identity/model required')
    incoming, outgoing = tokens(input_token_bound), tokens(max_output_tokens)
    if outgoing == 0:
        raise BudgetError('positive output bound required')
    inp, out = amount(input_price), amount(output_price)
    cost = amount(dollars(incoming*inp + outgoing*out))
    with transaction() as data:
        if data.get('evaluation_dispatch_blocked'):
            raise BudgetError('evaluation dispatch blocked after quote overrun')
        if any(x.get('call_id') == call_id for x in data['reservations']):
            raise BudgetError('call ID already held; implicit retry refused')
        total, evaluation = totals(data)
        if total+cost > amount(data['cap_usd']) or evaluation+cost > amount(data['allocations_usd']['evaluation']):
            raise BudgetError('call would exceed cumulative evaluation/shared cap')
        record = {'phase': 'evaluation', 'call_id': call_id, 'model': model,
                  'reserved_usd': dollars(cost), 'original_ceiling_usd': dollars(cost),
                  'input_token_bound': incoming, 'max_output_tokens': outgoing,
                  'input_price_usd_per_token': str(inp), 'output_price_usd_per_token': str(out),
                  'status': 'held-before-dispatch', 'held_unix': time.time(),
                  'actual_invoice_usd': None}
        data['reservations'].append(record)
    return record

def settle(call_id, usage_cost=None, usage=None,reconciliation_basis=None,currency_reported=None):
    """Reconcile known provider usage; retain entire hold when usage is absent."""
    with transaction() as data:
        matches = [x for x in data['reservations'] if x.get('call_id') == call_id]
        if len(matches) != 1:
            raise BudgetError('unknown or ambiguous call hold')
        record = matches[0]
        if record.get('status') == 'usage-reconciled':
            raise BudgetError('call already reconciled')
        usage = usage or {}
        basis = None
        known = None
        if usage_cost is not None:
            known = amount(usage_cost)
            basis = 'provider-reported usage cost'
        elif usage.get('cost') is not None:
            known = amount(usage['cost'])
            basis = 'provider-reported usage cost'
        elif usage.get('prompt_tokens') is not None and usage.get('completion_tokens') is not None:
            incoming, outgoing = tokens(usage['prompt_tokens']), tokens(usage['completion_tokens'])
            known = incoming*amount(record['input_price_usd_per_token']) + outgoing*amount(record['output_price_usd_per_token'])
            basis = 'reported token usage at frozen route prices'
            if incoming > record['input_token_bound'] or outgoing > record['max_output_tokens']:
                record['token_bound_violation'] = True
                data['evaluation_dispatch_blocked'] = 'reported usage exceeded token bound'
        if known is None:
            record.update(status='unknown-usage-hold-retained', settled_unix=time.time())
        else:
            record.update(reserved_usd=dollars(known), usage_cost_usd=dollars(known),
                          reconciliation_basis=reconciliation_basis or basis, currency_cost_provider_reported=currency_reported if currency_reported is not None else basis=='provider-reported usage cost', settled_unix=time.time(), status='usage-reconciled')
            if known > amount(record['original_ceiling_usd']):
                record['quote_overrun'] = True
                # Persist known excess so later calls cannot spend a fictitious balance.
                data['evaluation_dispatch_blocked'] = 'provider usage exceeded pre-call ceiling'
    return record

def snapshot():
    """Read current holds without changing ledger or calling any provider."""
    data = read_ledger()
    total, evaluation = totals(data)
    return {'shared_cap_usd': float(amount(data['cap_usd'])), 'accounted_shared_usd': float(total),
            'evaluation_cap_usd': float(amount(data['allocations_usd']['evaluation'])), 'accounted_evaluation_usd': float(evaluation),
            'remaining_evaluation_usd': float(max(Decimal(0), min(amount(data['allocations_usd']['evaluation'])-evaluation, amount(data['cap_usd'])-total))),
            'actual_invoice_usd': None}


def hold_resource(call_id,ceiling_usd,metadata):
    """Reserve one reviewed GPU envelope before the operator launches the job; no API call."""
    cost=amount(dollars(ceiling_usd))
    seconds=tokens(metadata.get('max_wall_seconds'))
    rate=amount(metadata.get('rate_usd_per_second'))
    fixed=amount(metadata.get('fixed_cost_usd',0))
    if not call_id or not metadata.get('app_name') or not seconds or rate<Decimal('0.00062992'):
        raise BudgetError('resource identity and finite conservative envelope required')
    if cost<=0 or cost<amount(dollars(seconds*rate+fixed)):
        raise BudgetError('resource hold below complete wall/resource envelope')
    with transaction() as data:
        if data.get('evaluation_dispatch_blocked'):raise BudgetError('evaluation dispatch blocked')
        if any(x.get('call_id')==call_id for x in data['reservations']):raise BudgetError('resource already held')
        total,evaluation=totals(data)
        if total+cost>amount(data['cap_usd']) or evaluation+cost>amount(data['allocations_usd']['evaluation']):
            raise BudgetError('resource would exceed shared/evaluation allocation')
        record={'phase':'evaluation','call_id':call_id,'model':'modal/L40S','reservation_type':'resource-envelope',
            'reserved_usd':dollars(cost),'original_ceiling_usd':dollars(cost),'status':'held-before-dispatch',
            'metadata':metadata,'held_unix':time.time(),'actual_invoice_usd':None}
        data['reservations'].append(record)
    return record

def settle_batch(call_ids,usage_cost,batch_id,reconciliation_basis=None,currency_reported=True):
    """Reconcile authoritative aggregate batch cost atomically; not item invoices."""
    if not call_ids or len(set(call_ids))!=len(call_ids) or not batch_id:raise BudgetError('batch identity required')
    known=amount(usage_cost)
    with transaction() as data:
        records=[]
        for ident in call_ids:
            matches=[x for x in data['reservations'] if x.get('call_id')==ident]
            if len(matches)!=1:raise BudgetError('unknown/ambiguous batch hold')
            records.append(matches[0])
        if all(x.get('batch_id')==batch_id and x.get('status')=='usage-reconciled' for x in records):return records
        if any(x.get('status')=='usage-reconciled' for x in records):raise BudgetError('partially/otherwise reconciled batch')
        ceiling=sum((amount(x['original_ceiling_usd']) for x in records),Decimal(0))
        if not ceiling:raise BudgetError('zero batch ceiling')
        remaining=known
        for index,record in enumerate(records):
            share=remaining if index==len(records)-1 else min(remaining,amount(dollars(known*amount(record['original_ceiling_usd'])/ceiling)))
            remaining-=share
            record.update(reserved_usd=dollars(share),allocated_batch_cost_usd=dollars(share),batch_id=batch_id,
                aggregate_batch_usage_cost_usd=dollars(known),status='usage-reconciled',settled_unix=time.time(),
                currency_cost_provider_reported=currency_reported,
                reconciliation_basis=reconciliation_basis or 'provider aggregate batch cost allocated by item ceilings; not per-item usage/invoice')
        if known>ceiling:data['evaluation_dispatch_blocked']='batch aggregate cost exceeded held ceiling'
    return records


def hold_batch(quotes):
    """Atomically reserve the complete batch item list before submission."""
    if not quotes or len({x['call_id'] for x in quotes})!=len(quotes):raise BudgetError('unique nonempty batch items required')
    prepared=[]
    for quote in quotes:
        incoming,outgoing=tokens(quote['input_token_bound']),tokens(quote['max_output_tokens'])
        if not quote['call_id'] or not quote['model'] or not outgoing:raise BudgetError('invalid batch item identity/output')
        inp,out=amount(quote['input_price']),amount(quote['output_price'])
        ceiling=dollars(incoming*inp+outgoing*out)
        prepared.append({'phase':'evaluation','call_id':quote['call_id'],'model':quote['model'],
            'reserved_usd':ceiling,'original_ceiling_usd':ceiling,'input_token_bound':incoming,'max_output_tokens':outgoing,
            'input_price_usd_per_token':str(inp),'output_price_usd_per_token':str(out),'status':'held-before-dispatch',
            'held_unix':time.time(),'actual_invoice_usd':None})
    cost=sum((amount(x['reserved_usd']) for x in prepared),Decimal(0))
    with transaction() as data:
        if data.get('evaluation_dispatch_blocked'):raise BudgetError('evaluation dispatch blocked')
        existing={x.get('call_id') for x in data['reservations']}
        if any(x['call_id'] in existing for x in prepared):raise BudgetError('batch item already held; no implicit retry')
        total,evaluation=totals(data)
        if total+cost>amount(data['cap_usd']) or evaluation+cost>amount(data['allocations_usd']['evaluation']):
            raise BudgetError('complete batch would exceed shared/evaluation allocation')
        data['reservations'].extend(prepared)
    return prepared

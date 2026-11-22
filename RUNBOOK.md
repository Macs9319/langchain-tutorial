# Pedalwise runbook

What to do when an alert from [`30_alerts.py`](30_alerts.py) arrives. Each
section is named after a rule in [`pedalwise/alerts.py`](pedalwise/alerts.py),
and every alert message links to its section.

**Urgent** alerts mean customers are affected now: start straight away.
**Warnings** can wait for working hours, but not for next week.

Commands below are for the single-copy Docker stack. For the scaled stack,
add `-f compose.scale.yaml` to each `docker compose`.

Every incident, whatever the alert:

1. Say in the team channel that you're on it, so two people don't fix the
   same thing in different ways.
2. Note the time things started, and what you did when. You'll want it for
   the write-up.
3. Afterwards, write down what happened, how you found out and how long
   that took, and what will stop it happening again: a test, an eval case,
   or a better alert.

## app_up

*Urgent.* `/readyz` didn't answer, twice in a row, from the alert checker's
side. Customers probably can't use the chat.

- `docker compose ps`: is `pedalwise` running, restarting, or unhealthy?
- `docker compose logs --tail 100 pedalwise`: a crash at startup is often a
  setting (`problems()` refuses unsafe production config, Tutorial 21) or a
  schema that's behind (Tutorial 28: run `28_migrations.py --upgrade`).
- `/readyz` answering 503 means the app runs but can't read its database:
  see [database](#database).
- If it's up from the server but not from the checker, look at what's in
  between: the network, Caddy (scaled stack), a firewall.
- A bad deploy? Go back to the previous image, then investigate.

Behind Caddy (scaled stack) this alert fires only when *every* copy is down:
Caddy routes around one dead copy. Look at `docker compose ps` for copies
that keep restarting.

## database

*Urgent.* The checker can't connect to the database, or can't read it.
Everything else depends on it: the other rules show as "unknown" until it's
back.

- SQLite: does the file exist at `PEDALWISE_DB`? Is the `pedalwise-data`
  volume mounted? Is the disk full (`df -h`)?
- Postgres: `docker compose -f compose.scale.yaml ps postgres` and its logs.
  Too many connections? Disk full?
- If data is damaged or gone: stop the app and restore the newest good
  backup (`29_backups.py --list`, then `--restore NAME`). Then compare the
  refund ledger with the payment provider: refunds after the backup are
  missing from it, and could be paid twice (Tutorial 29).

## error_rate

*Urgent.* More than 10% of chat turns in the last hour ended in an error
or a timeout (with at least 10 turns).

- `docker compose logs pedalwise | grep "Chat failed"`: the traceback says
  which part failed.
- Usually the model provider: an outage, a revoked or out-of-credit key, or
  rate limits. Check the provider's status page. The fallback model
  (Tutorial 16) should cover a primary outage: is `FALLBACK_MODEL` set, and
  does its key work?
- Timeouts but no errors: see [slow_replies](#slow_replies).
- Just deployed? Roll back first, investigate second.

The rate covers the last hour, so this stays firing for up to an hour after
the fix. Watch the newest runs on `/staff` to see whether it worked.

## slow_replies

*Warning.* One reply in 20 took longer than 30 seconds over the last hour.

- `22_observability.py --report --hours 1`: more model calls per run than
  usual means the agent is looping (a tool failing, then retried).
- The provider may just be slow: compare with its status page.
- With LangSmith tracing on (Tutorial 22), open a slow run by its id to see
  where the time went.

## token_spend

*Warning.* More than 1,000,000 tokens in the last hour. That's money: find
out why today.

- A burst of traffic, or one account sending a lot? Look at the `runs`
  table by customer.
- Many model calls per run means a loop; see [slow_replies](#slow_replies).
- If one account is abusing the chat, disable it: `27_auth.py
  --disable-user EMAIL` signs it out everywhere (Tutorial 28).
- Set a spending limit at the provider too. It's the backstop if this alert
  is missed.

## thumbs_down

*Warning.* More than half of the answers rated in the last hour got a
thumbs-down (with at least 5 ratings).

- `22_observability.py --export-evals FILE`: read what customers said.
- Did the documents in `data/` change, or the model (`CHAT_MODEL`)? Run
  `25_model_selection.py --cases FILE` on the cases you exported.
- Add the bad answers as eval cases, so the fix stays fixed.

## approvals_waiting

*Warning.* A refund has waited for a staff member for more than 4 hours.
The customer is waiting too.

- Someone with a staff account should open `/staff` and decide.
- If nobody can, that's a rota problem, not a software one: who covers
  approvals out of hours?

## backups

*Warning.* There's no good backup from the last 26 hours, or the newest
backup failed its verification. Nothing is lost yet, but the next disaster
would lose more than a day.

- `docker compose logs --tail 50 backup`: is the backup service running,
  and what did the last run say?
- A `.failed` file in the backup folder: `29_backups.py --verify NAME`
  shows why. Damaged disk? Database problem?
- Out of disk space? Prune old backups, or give them more room.
- Take one now and check it: `docker compose exec backup python
  29_backups.py --backup`.

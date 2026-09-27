# reldfs: install, test, fetch the data and reproduce the recorded runs (from the repo root).
# PY = a Python with reldfs and the harness requirements (README "Reproducing the numbers").
PY ?= .venv/bin/python
RUN = nice -n 10 $(PY)
TASKS = rel-f1/driver-dnf rel-f1/driver-top3 rel-trial/study-outcome rel-event/user-repeat \
        rel-event/user-ignore rel-avito/user-visits rel-avito/user-clicks rel-stack/user-engagement \
        rel-hm/user-churn rel-stack/user-badge rel-amazon/item-churn rel-amazon/user-churn

.PHONY: install test fetch leakcheck select final check-hashes zip test-run

install:            ## the package, the pinned versions and the test/harness extras
	$(PY) -m pip install -r harness/requirements.txt ".[test,harness]"

test:               ## synthetic-schema leak and determinism tests, selection-rule tests
	$(PY) -m pytest -q

fetch:              ## slim mirrors of the 7 datasets (board tasks only) under $$RELDFS_DATA (default data/)
	$(RUN) harness/fetch.py rel-f1 --tasks driver-dnf,driver-top3
	$(RUN) harness/fetch.py rel-trial --tasks study-outcome
	$(RUN) harness/fetch.py rel-event --tasks user-repeat,user-ignore
	$(RUN) harness/fetch.py rel-avito --tasks user-visits,user-clicks
	$(RUN) harness/fetch.py rel-hm --tasks user-churn
	$(RUN) harness/fetch.py rel-stack --tasks user-engagement,user-badge
	$(RUN) harness/fetch.py rel-amazon --tasks user-churn,item-churn

leakcheck:          ## leak check on all 12 tasks, one task per process (amazon peaks near 4.5 GB)
	for t in $(TASKS); do $(RUN) harness/leakcheck.py --tasks $$t --k 8 || exit 1; done

select:             ## the recorded H13b selection from the recorded backtest lines
	$(PY) harness/famselect.py c13_fam:b0 results/grid/h13_*.bt results/grid/h13b_*.bt results/grid/h13b-hm-rerun.bt

final:              ## full-size pass, one task per process: preds/final-{reldfs,h13,base}/ (about 3.5 h)
	for t in $(TASKS); do RELDFS_NOCACHE_WRITE=1 $(RUN) harness/finalpass.py --task $$t || exit 1; done

check-hashes:       ## compare preds/final-*/ with results/final-preds.sha256.tsv
	$(PY) harness/check_hashes.py

zip:                ## submissions/reldfs-classification.zip from preds/final-reldfs/ (no scoring)
	$(PY) harness/finaltest.py --final reldfs --package-only

test-run:           ## SCORES TEST (downloads the hosted labels): the report-only test run and the zip
	$(PY) harness/finaltest.py --final reldfs --supp h13,base

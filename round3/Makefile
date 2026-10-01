# reldfs round 3 ("r3"): reproduce the round-3 system from the recorded logs and the data.
# Run from this directory (round3/). PY = a Python with ../harness/requirements.txt and
# requirements.txt installed. Data: $RELDFS_DATA if set, else data/ here (or a symlink to the
# repo root's data/).
PY ?= ../.venv/bin/python
RUN = nice -n 10 $(PY)
export TABPFN_MODEL_CACHE_DIR ?= $(or $(RELDFS_DATA),data)/tabpfn
D = base+domonly+era+noid+robust
CHANGED = rel-f1/driver-top3 rel-f1/driver-dnf rel-trial/study-outcome rel-event/user-ignore rel-amazon/item-churn

.PHONY: text select gate leakcheck final copy-unchanged hashes zip

text:               ## rel-trial registration text for the point-in-time text score (~85 MB)
	$(RUN) harness/fetch_trial_text.py

select:             ## the per-task choice of rounds 2 and 3 from the recorded rolling-fold logs
	$(PY) harness/rbselect.py results/logs/r2/rb*.log results/logs/r3/rb*.log

gate:               ## the combined keep gate (12-task val mean gain >= .005 on seeds 0 and 1)
	$(PY) harness/r2gate.py --choice "rel-f1/driver-top3=$(D)+pfn" "rel-f1/driver-dnf=$(D)~$(D)+pfnk64" \
	  "rel-trial/study-outcome=base+txt|base+txt+robust~base+txt+pfnk64" "rel-event/user-ignore=base+dom+noid+robust" \
	  "rel-amazon/item-churn=base+hlq16" -- results/logs/r2/rb*.log results/logs/r3/rb*.log

leakcheck:          ## leak check of the changed tasks with the final code
	$(RUN) harness/leakcheck.py --cand c16_r3 --tasks rel-f1/driver-dnf,rel-f1/driver-top3,rel-trial/study-outcome,rel-event/user-ignore
	$(RUN) harness/leakcheck.py --cand c16_r3 --tasks rel-amazon/item-churn

final:              ## final pass of the 5 changed tasks -> preds/final-r3/ (5 seeds, refit on train+val)
	$(RUN) harness/finalpass3.py --cand c16_r3 --system r3 --task rel-f1/driver-top3 --variant "$(D)+pfn"
	$(RUN) harness/finalpass3.py --cand c16_r3 --system r3 --task rel-f1/driver-dnf --variant "$(D)~$(D)+pfnk64"
	$(RUN) harness/finalpass3.py --cand c16_r3 --system r3 --task rel-trial/study-outcome --variant "base+txt|base+txt+robust~base+txt+pfnk64"
	$(RUN) harness/finalpass3.py --cand c16_r3 --system r3 --task rel-event/user-ignore --variant "base+dom+noid+robust"
	RELDFS_NOCACHE_WRITE=1 $(RUN) harness/finalpass2.py --cand c16_r3 --system r3 --task rel-amazon/item-churn --variant base+hlq16

copy-unchanged:     ## the 7 unchanged tasks are the round-1 system's CSVs (make final at the repo root)
	for t in rel-event__user-repeat rel-avito__user-visits rel-avito__user-clicks rel-hm__user-churn \
	  rel-stack__user-engagement rel-stack__user-badge rel-amazon__user-churn; do \
	  cp ../preds/final-reldfs/$$t.csv preds/final-r3/ || exit 1; done

hashes:             ## compare preds/final-r3/ with results/final-r3.sha256.tsv
	cd preds/final-r3 && sha256sum -c ../../results/final-r3.sha256.tsv

zip:                ## submissions/reldfs-classification.zip from preds/final-r3/ (no scoring)
	$(PY) harness/finaltest.py --final r3 --package-only

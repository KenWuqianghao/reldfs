# Board plan: producing a full 12-task classification submission

The RelBench classification board counts only complete entries: predictions for all 12
tasks, from one system. This file is the procedure that produced them, with the measured cost
of each step, so that the run can be repeated. It was first written when 10 of the 12 tasks
ran end to end (step 3c in EXPERIMENTS.md) and was carried out in steps 7c-7d.

## 1. Data: slim mirrors by remote column reads

    df -h .                                   # the 7 mirrors take about 0.6 GB (amazon 136 MB, avito 179 MB)
    python harness/fetch.py rel-f1 --tasks driver-dnf,driver-top3
    python harness/fetch.py rel-trial --tasks study-outcome
    python harness/fetch.py rel-event --tasks user-repeat,user-ignore
    python harness/fetch.py rel-avito --tasks user-visits,user-clicks
    python harness/fetch.py rel-hm --tasks user-churn
    python harness/fetch.py rel-stack --tasks user-engagement,user-badge
    python harness/fetch.py rel-amazon --tasks user-churn,item-churn

- fetch.py reads each parquet file over HTTP range requests and drops string columns of 5 MB
  or more (compressed). On the Hub, rel-amazon's review.parquet is 7,201.7 MB,
  product.parquet 206.2 MB and customer.parquet 27.1 MB. The read drops review_text and
  summary (review), brand, title and description (product) and customer_name; review becomes
  about 123 MB and the rel-amazon mirror 136 MB (47 s).
- Dropped elsewhere: AdsInfo.Title (avito), customer.postal_code (hm), Body, Title, Text and
  AboutMe (stack).
- The mirror's test.parquet has no target column.
- Task table sizes (rows): amazon user-churn train 4,708,383 / val 409,792 / test 351,885;
  item-churn 2,536,014 / 177,689 / 166,842; hm user-churn 3,832,692 / 76,556 / 74,575;
  stack engagement 1,360,850 / 85,838 / 88,137 and badge 3,386,276 / 247,398 / 255,360;
  avito user-visits 86,619 / 29,979 / 36,129 and user-clicks 59,454 / 21,183 / 47,996.

## 2. Leak check before any number

    python harness/leakcheck.py --tasks <all 12, comma-separated> --k 8

Run the amazon tasks in a process of their own (peak about 4.5 GB). The check of the final
engine passed on all 12 tasks (86 cutoffs).

## 3. Final full-size pass

One process per task (harness/finalpass.py). Train is sampled to 1M rows at random
(RandomState(0)); hm, stack engagement and badge and both amazon tasks are capped, the other
tasks use every train row. LightGBM runs on one thread with early stopping on val, then a
refit on train + val for round(best_iter x (1 + n_val / n_train)) rounds; the CSV is the
mean of seeds 0 and 1.

    RELDFS_NOCACHE_WRITE=1 nice -n 10 python harness/finalpass.py --task rel-f1/driver-dnf --record

Measured cost of the recorded pass (single processes, LightGBM 1 thread, duckdb 2 threads,
on a shared 4-core machine at load 3-8; the times cover three systems' fits, the final one
and two supplementary ones):

| task(s) | seconds | peak RSS |
|---|---|---|
| f1 dnf, top3 | 70, 91 | 0.69, 0.39 GB |
| trial study-outcome | 203 | 1.91 GB |
| event user-repeat, user-ignore | 35, 107 | 2.35 GB |
| avito user-visits, user-clicks | 1022, 866 | 4.13, 4.23 GB |
| stack user-engagement, user-badge | 1135, 1901 | 2.69, 4.22 GB (a first badge run peaked at 5.02 GB before a memory fix, EXPERIMENTS.md 7c) |
| hm user-churn | 1622 | 3.55 GB |
| amazon item-churn, user-churn | 1957, 3661 | 4.16, 3.79 GB |

About 3.5 h of process time; 3.3 h of wall time on two lanes. Run hm, badge and the amazon
tasks alone; never run two processes above 4 GB side by side on an 8 GB machine.

## 4. Test scoring and the zip

    python harness/finaltest.py --final reldfs --supp h13,base --record

This is a test run: relbench.submit.evaluate_submission scores every CSV with
evaluate_task, the function the board's CI runs, against the hosted test tables, and then
relbench.submit's packager writes `submissions/reldfs-classification.zip` (12 CSVs,
11,377,079 bytes, under GitHub's 25 MB attachment limit). The recorded run happened once,
after the final system had been fixed and its CSVs hashed (results/final-preds.sha256.tsv).

To rebuild the zip from existing CSVs without scoring anything:

    python harness/finaltest.py --final reldfs --package-only

## 5. The issue form

https://github.com/stanford-star/relbench/issues/new?template=submit.yml has the fields Name,
Note, In-context? (Yes/No), URL and File(s). CI rescores the attached tables and posts the
report as a comment; a maintainer adds the `accept` label. The values used for reldfs are in
[submission.md](submission.md).

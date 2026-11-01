---
name: dataset-audit
description: Use when preparing or auditing Smart Product Search dataset files, provenance, catalog deduplication, or train/test boundaries.
---

# Dataset audit

Audit real source data against [DATA_SETUP.md](../../../docs/DATA_SETUP.md).
Read [AGENTS.md](../../../AGENTS.md) and the authorized iteration in
[IMPLEMENTATION_PLAN.md](../../../docs/IMPLEMENTATION_PLAN.md) first.

## Required inputs

- Authorized iteration and requested audit scope.
- Local dataset paths and source manifest, if available.
- Source revision, expected schema, and published split definitions.

## Procedure

1. Inspect existing scripts, manifests, and a few actual records before choosing
   checks. If inputs or scripts are absent, record the missing prerequisite;
   do not download data or implement a later iteration to complete an audit.
2. Check source identity and exact revision. For the approved English Small repack,
   require 419,653 train rows and 181,701 test rows. Stop on a count discrepancy;
   retain the observed counts and revision without substituting another dataset.
3. Check schema, locale `us`, `small_version == 1`, E/S/C/I label domain,
   null IDs, native/row split agreement, missing listing fields, duplicates,
   distinct queries, and query overlap between train and test.
4. Separate judgment and catalog checks. Preserve every judgment, deduplicate the
   catalog by `(product_locale, product_id)`, and measure metadata conflicts before
   applying a documented deterministic policy. Verify gold labels and query-product
   judgments cannot enter serving records, index text, or model input.
5. Check repeatability and, only if an index exists in the authorized iteration,
   catalog/index count agreement. Keep downloaded data and credentials out of Git.

## Outputs and verification

Store compact measured findings in `.artifacts/dataset-audit/audit.json` and a short
`audit.md`: commands, revision/checksums, actual counts, missingness, distributions,
conflicts, decisions, failures, and skipped checks with reasons. Preserve raw data.

An audit passes only after the required checks actually execute on complete data.
During Iteration 0, missing data is an expected prerequisite, not a successful
dataset audit. Never replace missing measurements with published counts.

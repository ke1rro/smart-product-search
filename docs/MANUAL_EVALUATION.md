# Manual evaluation of the complete search workflow

I evaluate five fixed shopping queries using BM25 followed by **Analyze with AI**.
I will keep the current model, prompt, and settings unchanged throughout the check.
This is a small convenience sample for manual inspection, not a representative
benchmark. Some queries have also appeared in development examples.

I reduced the original 10-query plan to its first five queries after reviewing
Q01-Q04. The query texts, order, and scoring rules remain unchanged. This scope
adjustment occurred during evaluation to reduce the workload.

The reference labels and explanation ratings are preliminary assessments proposed
with LLM assistance. I will review them and report unresolved disagreements.
Human calibration is pending, so these are not independent human reference labels.

## Fixed queries

I will keep every query below, including unsuccessful cases, instead of replacing
queries after seeing the results.

| ID  | Case                        | Exact query                            |
| --- | --------------------------- | -------------------------------------- |
| Q01 | General product request     | `wireless noise cancelling headphones` |
| Q02 | Capacity and intended use   | `waterproof hiking backpack 35 liter`  |
| Q03 | Accessory and compatibility | `case for Sony WH-1000XM4 headphones`  |
| Q04 | Color and material          | `white cotton t shirt`                 |
| Q05 | Material and measurement    | `stainless steel water bottle 1 liter` |

These requests do not guarantee that matching products exist in the catalog.

## Procedure

1. Record the date, application version or commit, configured model, and prompt
   version. Use **BM25** for all five queries.
2. Submit the exact query with **Search**. Save the top 10 product IDs in retrieval
   order, or all products if fewer are returned. Save their available listing text.
3. Before selecting **Analyze with AI**, assign my reference label to each product
   using the query and listing metadata. Use `U` when evidence is insufficient.
   Do not consult official ESCI labels.
4. Select **Analyze with AI** once. Record the predictions and explanations by
   product ID, because grouping changes the display order. Record whether the
   response came from the API, cache, or was unavailable.
5. Check each explanation against the available metadata. Save the displayed
   output or screenshots alongside my ratings. Reuse successful cached responses
   instead of requesting new predictions for identical inputs.

## Rating rules

| Rating | My interpretation                                                                   |
| ------ | ----------------------------------------------------------------------------------- |
| E      | The metadata supports the requested product type and important stated requirements. |
| S      | A reasonable replacement serving the same purpose, with a relevant difference.      |
| C      | A product used with the requested item, rather than a replacement.                  |
| I      | The product does not satisfy or complement the request.                             |
| U      | The metadata is insufficient for a defensible reference label.                      |

`U` is a manual uncertainty marker, not an additional model category. Missing
metadata does not establish that a requirement is satisfied or violated.

For explanation support, I will use **Yes** when every factual claim is supported,
**No** when a claim is contradicted or invented, and **U** when I cannot judge it.
I will distinguish model errors from incomplete listing information.

## Product ratings

Copy the row for every actual product in each top-10 batch. Leave predictions empty
when analysis is unavailable, and record the failure separately.

The [50 recorded product ratings](manual_evaluation_ratings.csv) contain the five
completed preliminary batches. Reference labels, predictions, exact explanations,
and assessment notes are retained separately. Author review remains pending.

| Query ID | Rank | Product ID | My label | Predicted label | Explanation supported, Yes/No/U | Evidence or error note |
| -------- | ---: | ---------- | -------- | --------------- | ------------------------------- | ---------------------- |
| Q01      |      |            |          |                 |                                 |                        |

## Query outcomes

**Useful candidate** means at least one product in the retrieved top 10 has my
reference label E or S. **Final success** requires at least one such product to
appear in the predicted E/S groups with a supported explanation. Record
`Unavailable` when classification fails and `U` when the outcome cannot be judged.

| Query ID | Retrieved candidates checked | Useful candidate, Yes/No/U | Final success, Yes/No/U/Unavailable | API/cache/unavailable |
| -------- | ---------------------------: | -------------------------- | ----------------------------------- | --------------------- |
| Q01      |                           10 | Yes                        | Yes                                 | API                   |
| Q02      |                           10 | Yes                        | Yes                                 | Cache                 |
| Q03      |                           10 | No                         | No                                  | API                   |
| Q04      |                           10 | Yes                        | Yes                                 | API                   |
| Q05      |                           10 | Yes                        | Yes                                 | API                   |

These outcomes are preliminary LLM-assisted assessments. Q02 passes the E/S rule
through substitutes, without a confirmed waterproof 35 L Exact result. For Q05,
I accepted nominal 32oz/945ml as approximately one liter before predictions were
provided. The application still received the original query without that clarification.

| Query | Category matches / scorable products | Supported explanations |
| ----- | -----------------------------------: | ---------------------: |
| Q01   |                                  4/8 |                   4/10 |
| Q02   |                                  3/3 |                   9/10 |
| Q03   |                                 7/10 |                  10/10 |
| Q04   |                                 9/10 |                   9/10 |
| Q05   |                                10/10 |                  10/10 |
| Total |                        33/41, 80.49% |             42/50, 84% |

Nine U references are excluded from category agreement, leaving 41/50 products
scorable. All 50 explanations were rated. Two C/I decisions for Q03 require review.
Pooled agreement is not accuracy against official ESCI labels. No provider request
latency, token usage or HTTP-attempt counts were supplied for this walkthrough.

## Measurements to report

- **Useful candidate@10:** queries rated Yes for a useful candidate, divided by 5.
- **Final task success:** queries rated Yes for final success, divided by 5.
- **Category agreement:** matching reference and predicted labels, divided by
  products with both a non-U reference and a valid prediction.
- **Explanation support:** explanations rated Yes, divided by explanations rated
  Yes or No. Report U counts and missing outputs separately.
- **Coverage and failures:** actual products rated, classified batches, API calls,
  cache hits, unavailable batches, and all uncertain ratings.

Keep the five-query denominators for query-level measurements. Uncertain cases do
not count as confirmed successes, but must be reported separately. These scores
are observations on this sample, not exhaustive catalog recall. I will include
two or three concrete errors and state who assigned and reviewed the ratings.

## Separate behavior checks

| Input or condition                           | Expected behavior                                                                              | Observed result |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------- | --------------- |
| Empty query                                  | Guidance to enter a query, with no classification request                                      |                 |
| `zzzxqvnotarealproduct987654321`             | Honest empty state if there are no lexical matches                                             |                 |
| Unavailable API with no matching cache entry | Retrieved products remain visible, with an explicit unavailable warning and no invented labels |                 |

Check API failure using the existing mocked UI tests rather than changing a real
credential. Cache hits can still succeed without API access. Keep behavior checks
separate from the relevance and explanation scores above.

Do not claim that I completed this evaluation until the tables contain actual
observations. In the report, state the sample size, scored denominators, uncertainty
counts, measured results, and limitations.

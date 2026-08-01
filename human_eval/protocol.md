# Human evaluation protocol

## Purpose
Automatic metrics (BLEU, ROUGE-L, lexical grounding) and the LLM judge are
proxies. This protocol collects expert judgements from HR professionals on the
dimensions that matter for a statute-grounded assistant: accuracy, citation
correctness, completeness, usefulness, and potential for harm.

## Raters
Target 3 raters with HR or labour-law experience in Bangladesh. Raters must not
have been involved in building the dataset or the model.

## Materials
- `rate.html` - the blind rating application. It is fully offline: open it in any
  browser, no installation or internet connection is required.
- Raters receive **only** `rate.html`. They must not receive `answer_key.json`.

## Blinding
Each question is shown with one answer per system under neutral labels
(A, B, C, ...). The label-to-system assignment is randomised independently for
every question, so position carries no information. The HTML contains no system
names, model names, or automatic scores. The mapping is stored separately in
`answer_key.json` and is joined to the ratings only after collection.

## Sample
38 questions sampled with a fixed seed (3407) from the scenario test set,
covering the HR topics represented in that set. Each rater rates every question,
so all raters see the same items (required for the agreement statistic).

## Rating scale
1 = unacceptable, 3 = adequate, 5 = excellent

| Criterion | Question put to the rater |
|---|---|
| Accuracy | Is the answer correct under the Bangladesh Labour Act 2006? |
| Citation correctness | Are the section numbers cited correct and relevant? (Score 1 if no section is cited but one was needed.) |
| Completeness | Does it fully address what was asked, without leaving out an essential condition or exception? |
| Usefulness | Could you act on this answer in day-to-day HR work? |
| Potentially harmful / misleading | Would following this answer risk a wrong or unlawful HR decision? (yes/no) |

The reference answer is available behind a collapsed control on each question.
Raters are instructed to form their own view first and open it only to check a
specific point, so that ratings are not anchored to the reference wording.

## Procedure
1. Rater enters an identifier, then works through the questions top to bottom.
2. Progress is saved to browser local storage, so the task can be done in
   several sittings on the same machine and browser.
3. On completion the rater clicks *Download CSV* and returns the file.

## Analysis
- Join each rater CSV to `answer_key.json` on `item` + `answer_label` to recover
  the system identity.
- Report the mean per criterion per system, with 95% confidence intervals over
  items (bootstrap, 10 000 resamples).
- Report inter-annotator agreement with Krippendorff's alpha (ordinal level) for
  the 1-5 criteria and nominal level for the harm flag. Report alpha alongside
  the means: means from raters who do not agree are not interpretable.
- Compare systems with a paired test over items (Wilcoxon signed-rank), since
  every system answers the same questions.

## Reporting
State the number of raters, their background, the agreement statistic, and the
sample size. Human ratings are reported separately from automatic metrics and
are never mixed into the same column.

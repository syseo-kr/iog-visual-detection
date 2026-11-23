# Release verification results

Executed successfully on the derived-data candidate (no APIs, no raw images):

- 2,519 rows, 1,300 IOG, 1,219 controls; 1,775 pHash groups.
- All full-fold pHash components remain intact.
- Corrected dedup representatives: one per component, 624 IOG + 1,151 control.
- 8 mixed-label components and 59 screenshots.
- 875 distinct raw-byte MD5 values among IOG screenshots.
- Original stored CLIP probe accuracy 94.0453%, and retrospective OR recall 95.0%.
- Original missed/recovered counts: 203/138/65.
- Main prompt experiment: 7,080 valid predictions; 570 common complete cases.
- Resolution experiment: 160 matched cases for each provider and three conditions.
- API latency and tokens: 96 successful ten-screenshot requests.
- Gemini input tokens: 49,680 under padded 224 and padded 1024.
- GPT reasoning tokens: no separate breakdown retained; output tokens used for cost.
- Exact pHash-cluster sign flip + Holm: 0.05010242 (CLIP vs Claude),
  0.05522320 (CLIP vs OR), no significance at alpha 0.05.
- CPU timing: five runs, median preprocessing plus inference 448.277 ms per image.

Note: `reproduce.py` contains computationally expensive full training and was not
run to completion in this preparation pass. The new scripts reproduce derived-data
checks and the supplementary analyses from the released offline predictions.
Bootstrap CI numerical endpoints can vary slightly from the paper when using a
new fixed seed (20261123) although the method is the same.

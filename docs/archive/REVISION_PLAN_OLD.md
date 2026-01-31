# CLPR论文大修计划 (Major Revision Plan)
# 邮件意见：
Thank you very much for considering JBHI to publish your research work.

Sincerely,

Prof. Dimitrios I. Fotiadis
Editor-in-Chief

Cc: file

Associate Editor's comments to the authors:
Associate Editor
Comments to the Author:
Dear All,
I think this manuscript presents an interesting approach.

Please address comments from the reviewers, their main comments are concerns around the practical significance of the addressed problem and the alignment between the stated motivation and the proposed solution, some comments / questions around the methodology used and better description of the performance metrics, its generalisability and comparison with previous presented models.  

Please consider improving Figures 4, 5 and 6 (making them clearer as at the moment using small character size, and very difficult to see).

Please ensure that if and when resubmitting the paper does not exceed the page limit.

For submitting your revised manuscript, please ensure all changes are clearly highlighted in the manuscript and explained in detail in the rebuttal to facilitate the review process.

Reviewers' comments to the authors:
Reviewer: 1

Comments to the Corresponding Author
Dear Authors,
The manuscript could be improved if following comments are taken into consideration:
1. Problem Motivation: The connection between the "information overload vs. evidence scarcity paradox" and personalized re-ranking needs clarification. The current solution optimizes ranking but doesn't directly address evidence quality or scarcity. Consider:
1) Explicitly showing how personalization helps identify rare but important studies
2) Adding quality indicators (study design, evidence level) to the ranking criteria
3) Demonstrating retrieval of typically hard-to-find evidence (negative results, non-English studies)

2. Experimental Limitations:
1) Conversation Length: Testing only 3-turn conversations is insufficient. Extend to 5-10 turns to validate scalability and profile drift
2) Baseline Coverage: Include comparisons with state-of-the-art LLMs (GPT-4, Claude, Gemini) and specialized tools (Elicit, Consensus)
3) Error Analysis: Add qualitative analysis of failure cases, identifying when and why CLPR fails to improve rankings


Reviewer: 2

Comments to the Corresponding Author
Summary:
This paper introduces the CLPR framework for conversational literature retrieval, utilizing an LLM to generate a dynamic, multi-dimensional personalized user profile (combining sequential, focus, and background cues) to improve re-ranking quality. The approach demonstrates strong performance on the novel MedCorpus and existing LitSearch datasets. While the core framework is sound and the contribution is clear, the evaluation lacks the experimental depth, latest SOTA comparison, and validation rigor, which ultimately reduces the overall rigor of the work.
Strengths:
The paper presents a sound and complete method with certain innovation. The CLPR framework is logically robust, and the introduction of a dynamic, multi-dimensional profile (capturing sequential, focus, and long-term background cues) via LLM generation represents a concrete and valuable contribution to conversational IR. The existing ablation study successfully validates the contribution of each component, and the creation of the MedCorpus dataset demonstrates strong cross-domain generalizability.
Weaknesses:
The experimental evaluation suffers from several issues that reduce the overall rigor and clarity of the work. The main issues include: insufficient comparison with modern SOTA baselines; the evaluation is incomplete for an interactive system, relying solely on offline relevance metrics; and the critical risk of profile inference errors and potential 'profile hallucination' is entirely undiscussed and unverified.
Concerns:
1.Insufficient Comparison with the Latest Baselines: The experimental evaluation lacks sufficient comparison with 2024 and 2025 SOTA personalized or LLM-enhanced conversational retrieval models. This addition is necessary to robustly validate the claimed superiority and timeliness of the CLPR framework.
2.Lack of Real User Validation for Interaction: As an inherently interactive and personalized system, the evaluation is currently limited to offline metrics (NDCG, Recall). The paper should acknowledge this limitation and either supplement the study with a small-scale user experiment (e.g., measuring satisfaction).
Profile Hallucination Requires Empirical Discussion: The reliance on LLM-generated profiles introduces a risk of "Profile Hallucination" (the LLM incorrectly infers or fabricates user intent). This risk must be addressed through supplementary experiments that quantitatively validate the factuality and robustness of the generated profile text.

Reviewer: 3

Comments to the Corresponding Author
This manuscript proposes a method for retrieval reranking leveraging user profile and query history. The study is overall well done and well reported. I have a few comments.

General comments:
1. The part of the title "Profile Re-ranking" indicates reranking of the profile rather than the retrieved documents. Suggest "Profile Based Re-ranking".
2. Several hyperparameters are not clearly explained, including the alpha in equation (2), the beta in equation (3), and gamma, tau, and k in Algorithm 1. While the beta in equation (5) is studied in the experiments, these other parameters are not discussed. Furthermore, using the same name in different equations also causes confusion (e.g. beta in multiple equations and k in different contexts).
3. The function CalculateContinuity should be further explained.
4. The metrics for performance evaluation of the "Initial Retrieval" are not reasonable. If reranking is not performed after retrieval, it does not make sense to retrieve 100 documents and then compute metrics such as NDCG@10 and MAP@10. I suggest computing the metrics by retrieving only 10 documents for the "Initial Retrieval".
5. For LitSearch in Fig. 6, FOC is shown to have effects. However, since both FOC and BG are derived from history H, why BG is not analyzed in the ablation studies for LitSearch? Furthermore, SEQ appears to have little effect on MedCorpus. Please discuss.

Specific comments:
1. [Page 2, Line 59, right column] "section II" --> "section III"?
2. [Page 7, Line 52, right column] Correct "reffig:ablation"
3. [Page 8, Line 52] "with no history" - Is this correct? Doesn't FOC make use of history?


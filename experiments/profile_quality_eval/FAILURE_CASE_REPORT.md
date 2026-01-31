# Failure Case Analysis Report

**Total failure cases found:** 19
**Top 19 cases shown below (sorted by severity)**

## Summary by Failure Type

- **top1_regression**: 8 cases
- **ndcg_regression**: 6 cases
- **other**: 4 cases
- **both_miss**: 1 cases

---

## Case 1: LitSearch - Query 105

**Failure Type:** `top1_regression`
**Severity Score:** 2.944

### Query
> Are there any studies on incorporating external commonsense knowledge into conversational models to enhance emotional support?

### Profile
> Research on artificial intelligence focusing on incorporating external commonsense knowledge into conversational models in context of enhancing emotional support through improved contextual understanding and empathetic responses.

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 0.00 | -1.00 |
| NDCG@10 | 1.000 | 0.431 | -0.569 |
| MRR@10 | 1.000 | 0.250 | -0.750 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `247748640` - MISC: A MIxed Strategy-Aware Model Integrating COMET for Emotional Support Conve...
2. [✗] `252780132` - Improving Multi-turn Emotional Support Dialogue Generation with Lookahead Strate...
3. [✗] `195069365` - Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset...
4. [✗] `259370732` - PAL to Lend a Helping Hand: Towards Building an Emotion Adaptive Polite and Empa...
5. [✗] `258740721` - Knowledge-enhanced Mixed-initiative Dialogue System for Emotional Support Conver...

**Personalized (Profile+Query):**

1. [✗] `253628217` - 基於常識知識的移情對話回覆生成 Improving Response Diversity through Commonsense-Aware Empatheti...
2. [✗] `258740721` - Knowledge-enhanced Mixed-initiative Dialogue System for Emotional Support Conver...
3. [✗] `216562921` - Towards Persona-Based Empathetic Conversational Models...
4. [✓] `247748640` - MISC: A MIxed Strategy-Aware Model Integrating COMET for Emotional Support Conve...
5. [✗] `259370732` - PAL to Lend a Helping Hand: Towards Building an Emotion Adaptive Polite and Empa...

### Ground Truth Relevant Documents

- `247748640` (rel=1)

---

## Case 2: LitSearch - Query 251

**Failure Type:** `top1_regression`
**Severity Score:** 2.833

### Query
> Could you suggest some work that develops multimodal models with contrastive learning approaches?

### Profile
> Research on Could you suggest some work that develops multimodal models with contrastive learning approaches?

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 0.00 | -1.00 |
| NDCG@10 | 1.000 | 0.500 | -0.500 |
| MRR@10 | 1.000 | 0.333 | -0.667 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `229924402` - UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contra...
2. [✗] `248377556` - MCSE: Multimodal Contrastive Learning of Sentence Embeddings...
3. [✗] `222310695` - Does my multimodal model learn cross-modal interactions? It's harder to tell tha...
4. [✗] `257557257` - Published as a conference paper at ICLR 2023 IDENTIFIABILITY RESULTS FOR MULTIMO...
5. [✗] `220302636` - Relating by Contrasting: A Data-efficient Framework for Multimodal Generative Mo...

**Personalized (Profile+Query):**

1. [✗] `257557257` - Published as a conference paper at ICLR 2023 IDENTIFIABILITY RESULTS FOR MULTIMO...
2. [✗] `248377556` - MCSE: Multimodal Contrastive Learning of Sentence Embeddings...
3. [✓] `229924402` - UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contra...
4. [✗] `252819166` - Modeling Intra-and Inter-Modal Relations: Hierarchical Graph Contrastive Learnin...
5. [✗] `220425360` - Contrastive Code Representation Learning...

### Ground Truth Relevant Documents

- `229924402` (rel=1)

---

## Case 3: MedCorpus - Query topic_095_1

**Failure Type:** `top1_regression`
**Severity Score:** 2.672

### Query
> Are there any known complications of SARS-CoV-2 infection that affect bone health?

### Profile
> Researcher in infectious disease and musculoskeletal health investigating SARS-CoV-2 infection complications affecting bone health in clinical contexts.

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 0.00 | -1.00 |
| NDCG@10 | 0.972 | 0.633 | -0.339 |
| MRR@10 | 1.000 | 0.333 | -0.667 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `permed-68512` - COVID-19...
2. [✓] `permed-55991` - Osteonecrosis and Osteomyelitis of the Proximal Third of Tibia as a Late Sequela...
3. [✓] `permed-91998` - Post-COVID-19 Myositis Based on Magnetic Resonance Imaging: A Case Report....
4. [✗] `permed-68513` - SARS-CoV-2...
5. [✓] `permed-54238` - Elevated plasma CAF22 are incompletely restored six months after COVID-19 infect...

**Personalized (Profile+Query):**

1. [✗] `permed-68513` - SARS-CoV-2...
2. [✗] `permed-56805` - SARS-CoV-2 infection and cardiac arrhythmias....
3. [✓] `permed-55991` - Osteonecrosis and Osteomyelitis of the Proximal Third of Tibia as a Late Sequela...
4. [✓] `permed-68512` - COVID-19...
5. [✓] `permed-51264` - SARS-CoV-2 Infection and Liver Disease: A Review of Pathogenesis and Outcomes....

### Ground Truth Relevant Documents

- `permed-55991` (rel=2)
- `permed-68512` (rel=2)
- `permed-91998` (rel=1)
- `permed-55998` (rel=1)
- `permed-51264` (rel=1)

---

## Case 4: MedCorpus - Query topic_148_2

**Failure Type:** `top1_regression`
**Severity Score:** 2.596

### Query
> Impact of high-pressure discovery of novel hydrate structures on understanding these deep-sea gas hydrate formation mechanisms

### Profile
> Research on Impact of high-pressure discovery of novel hydrate structures on understanding these deep-sea gas hydrate formation mechanisms

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 0.00 | -1.00 |
| NDCG@10 | 0.921 | 0.576 | -0.346 |
| MRR@10 | 1.000 | 0.500 | -0.500 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `permed-86233` - SciFinder...
2. [✗] `permed-50883` - Maleic and Methacrylic Homopolymers with Pendant Dibutylamine or Dibutylamine Ox...
3. [✓] `permed-73148` - Dynamic Dissociation Behaviors of sII Hydrates in Liquid Water by Heating: A Mol...
4. [✗] `permed-73153` - Inhibitory influence of amino acids on the formation kinetics of methane hydrate...
5. [✓] `permed-73149` - Fractal analysis on CO<sub>2</sub> hydrate-bearing sands during formation and di...

**Personalized (Profile+Query):**

1. [✗] `permed-73153` - Inhibitory influence of amino acids on the formation kinetics of methane hydrate...
2. [✓] `permed-73148` - Dynamic Dissociation Behaviors of sII Hydrates in Liquid Water by Heating: A Mol...
3. [✓] `permed-73149` - Fractal analysis on CO<sub>2</sub> hydrate-bearing sands during formation and di...
4. [✗] `permed-50883` - Maleic and Methacrylic Homopolymers with Pendant Dibutylamine or Dibutylamine Ox...
5. [✗] `permed-66975` - The genome of a hadal sea cucumber reveals novel adaptive strategies to deep-sea...

### Ground Truth Relevant Documents

- `permed-86233` (rel=2)
- `permed-73149` (rel=1)
- `permed-73148` (rel=1)
- `permed-75490` (rel=1)
- `permed-75488` (rel=1)

---

## Case 5: MedCorpus - Query topic_095_2

**Failure Type:** `top1_regression`
**Severity Score:** 2.506

### Query
> How might this impact on bone health be related to the virus's effects on the body's inflammatory response?

### Profile
> Researcher in immunology and musculoskeletal health investigating virus-induced inflammatory response impacts on bone health for clinical and pathological context.

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 0.00 | -1.00 |
| NDCG@10 | 0.905 | 0.649 | -0.256 |
| MRR@10 | 1.000 | 0.500 | -0.500 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `permed-68512` - COVID-19...
2. [✓] `permed-82274` - Viral infections in orthopedics: A systematic review and classification proposal...
3. [✓] `permed-74285` - Chebulinic acid alleviates LPS-induced inflammatory bone loss by targeting the c...
4. [✓] `permed-53858` - Pyroptosis in inflammatory bone diseases: Molecular insights and targeting strat...
5. [✗] `permed-26324` - 人类免疫缺陷病毒相关骨质疏松的诊断与治疗的研究进展...

**Personalized (Profile+Query):**

1. [✗] `permed-26324` - 人类免疫缺陷病毒相关骨质疏松的诊断与治疗的研究进展...
2. [✓] `permed-74285` - Chebulinic acid alleviates LPS-induced inflammatory bone loss by targeting the c...
3. [✓] `permed-68512` - COVID-19...
4. [✓] `permed-82274` - Viral infections in orthopedics: A systematic review and classification proposal...
5. [✓] `permed-53858` - Pyroptosis in inflammatory bone diseases: Molecular insights and targeting strat...

### Ground Truth Relevant Documents

- `permed-55991` (rel=2)
- `permed-68512` (rel=2)
- `permed-82274` (rel=1)
- `permed-72512` (rel=1)
- `permed-53858` (rel=1)

---

## Case 6: MedCorpus - Query topic_037_2

**Failure Type:** `top1_regression`
**Severity Score:** 2.367

### Query
> Impact of targeting this interaction on the tumor microenvironment of such cancers, focusing on changes in cytokine release and key immune cell populations

### Profile
> Researcher in oncology investigating impact of targeting this interaction on the tumor microenvironment of such cancers, focusing on changes in cytokine release and key immune cell populations.

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 0.00 | -1.00 |
| NDCG@10 | 0.899 | 0.782 | -0.117 |
| MRR@10 | 1.000 | 0.500 | -0.500 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `permed-67709` - Therapeutic targeting of LIF overcomes macrophage mediated immunosuppression of ...
2. [✗] `permed-50744` - The Role of Cytokinome in the HNSCC Tumor Microenvironment: A Narrative Review a...
3. [✓] `permed-68242` - Targeting the CD47/thrombospondin-1 signaling axis regulates immune cell bioener...
4. [✓] `permed-56649` - How the Tumor Micromilieu Modulates the Recruitment and Activation of Colorectal...
5. [✓] `permed-67788` - TAM-targeted reeducation for enhanced cancer immunotherapy: Mechanism and recent...

**Personalized (Profile+Query):**

1. [✗] `permed-50744` - The Role of Cytokinome in the HNSCC Tumor Microenvironment: A Narrative Review a...
2. [✓] `permed-67709` - Therapeutic targeting of LIF overcomes macrophage mediated immunosuppression of ...
3. [✓] `permed-68242` - Targeting the CD47/thrombospondin-1 signaling axis regulates immune cell bioener...
4. [✓] `permed-56649` - How the Tumor Micromilieu Modulates the Recruitment and Activation of Colorectal...
5. [✓] `permed-67788` - TAM-targeted reeducation for enhanced cancer immunotherapy: Mechanism and recent...

### Ground Truth Relevant Documents

- `permed-56649` (rel=1)
- `permed-67709` (rel=1)
- `permed-67788` (rel=1)
- `permed-53177` (rel=1)
- `permed-68242` (rel=1)

---

## Case 7: MedCorpus - Query topic_117_2

**Failure Type:** `top1_regression`
**Severity Score:** 2.363

### Query
> How effective is it in reducing complications when donors are not HLA-compatible, and what challenges does this pose?

### Profile
> Researcher in transplant medicine examining non-HLA-compatible donor-recipient matches for reducing complications and challenges in graft survival, immune rejection, and long-term outcomes.

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 0.00 | -1.00 |
| NDCG@10 | 0.699 | 0.586 | -0.113 |
| MRR@10 | 1.000 | 0.500 | -0.500 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `permed-75546` - HLA-DQ Mismatches Lead to More Unacceptable Antigens, Greater Sensitization, and...
2. [✓] `permed-82635` - HLA Factors Versus Non-HLA Factors for Haploidentical Donor Selection....
3. [✗] `permed-82637` - The clinical impact of donor against recipient HLA one way mismatch on the occur...
4. [✓] `permed-82636` - Clinical impact of KIR haplotypes in 10/10 HLA-matched unrelated donor-recipient...
5. [✗] `permed-54623` - Loss of heterozygosity leading to incorrect HLA typing for platelet-transfusion ...

**Personalized (Profile+Query):**

1. [✗] `permed-82637` - The clinical impact of donor against recipient HLA one way mismatch on the occur...
2. [✓] `permed-82635` - HLA Factors Versus Non-HLA Factors for Haploidentical Donor Selection....
3. [✓] `permed-75546` - HLA-DQ Mismatches Lead to More Unacceptable Antigens, Greater Sensitization, and...
4. [✓] `permed-82636` - Clinical impact of KIR haplotypes in 10/10 HLA-matched unrelated donor-recipient...
5. [✗] `permed-61456` - Corrigendum: Disqualification of donor and recipient candidates from the living ...

### Ground Truth Relevant Documents

- `permed-81962` (rel=2)
- `permed-82635` (rel=1)
- `permed-75546` (rel=1)
- `permed-47578` (rel=1)
- `permed-55936` (rel=1)

---

## Case 8: MedCorpus - Query topic_181_2

**Failure Type:** `top1_regression`
**Severity Score:** 2.330

### Query
> How does understanding the genetic diversity of these pathogens impact the development of resistant varieties of A. sinicus?

### Profile
> Research on How does understanding the genetic diversity of these pathogens impact the development of resistant varieties of a. sinicus?

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 0.00 | -1.00 |
| NDCG@10 | 0.711 | 0.631 | -0.080 |
| MRR@10 | 1.000 | 0.500 | -0.500 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `permed-78965` - Patterns of Genomic Variations in the Plant Pathogen <i>Dickeya solani</i>....
2. [✓] `permed-84400` - Genome-wide association study for resistance in bread wheat (Triticum aestivum L...
3. [✗] `permed-64032` - Genetic Diversity of <i>Candidatus</i> Liberibacter asiaticus Based on Four Hype...
4. [✓] `permed-85023` - Diversity Analysis of the Rice False Smut Pathogen <i>Ustilaginoidea virens</i> ...
5. [✗] `permed-88390` - Genomic Variation Underlying the Breeding Selection of Quinoa Varieties Longli-4...

**Personalized (Profile+Query):**

1. [✗] `permed-64032` - Genetic Diversity of <i>Candidatus</i> Liberibacter asiaticus Based on Four Hype...
2. [✓] `permed-85023` - Diversity Analysis of the Rice False Smut Pathogen <i>Ustilaginoidea virens</i> ...
3. [✓] `permed-78965` - Patterns of Genomic Variations in the Plant Pathogen <i>Dickeya solani</i>....
4. [✓] `permed-86092` - Variations in genetic diversity in cultivated <i>Pistacia chinensis</i>....
5. [✗] `permed-88390` - Genomic Variation Underlying the Breeding Selection of Quinoa Varieties Longli-4...

### Ground Truth Relevant Documents

- `permed-80024` (rel=2)
- `permed-85023` (rel=1)
- `permed-78965` (rel=1)
- `permed-86092` (rel=1)
- `permed-70022` (rel=1)

---

## Case 9: MedCorpus - Query topic_148_3

**Failure Type:** `ndcg_regression`
**Severity Score:** 0.218

### Query
> Influence of newly discovered high-pressure potassium chloride hydrate on deep-sea gas hydrate formation/stability and marine geochemical cycles

### Profile
> Research on Influence of newly discovered high-pressure potassium chloride hydrate on deep-sea gas hydrate formation/stability and marine geochemical cycles

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 1.00 | +0.00 |
| NDCG@10 | 0.914 | 0.695 | -0.218 |
| MRR@10 | 1.000 | 1.000 | +0.000 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `permed-86233` - SciFinder...
2. [✗] `permed-75492` - Hydrates of active pharmaceutical ingredients: A <sup>35</sup>Cl and <sup>2</sup...
3. [✓] `permed-75487` - Crystal structure of potassium chloride monohydrate: water intercalation into th...
4. [✓] `permed-75490` - Abnormal structural transformation of tetra-<i>n</i>-butyl ammonium chloride + X...
5. [✗] `permed-50883` - Maleic and Methacrylic Homopolymers with Pendant Dibutylamine or Dibutylamine Ox...

**Personalized (Profile+Query):**

1. [✓] `permed-75487` - Crystal structure of potassium chloride monohydrate: water intercalation into th...
2. [✗] `permed-91516` - Schneiderian...
3. [✓] `permed-73153` - Inhibitory influence of amino acids on the formation kinetics of methane hydrate...
4. [✓] `permed-86233` - SciFinder...
5. [✗] `permed-50883` - Maleic and Methacrylic Homopolymers with Pendant Dibutylamine or Dibutylamine Ox...

### Ground Truth Relevant Documents

- `permed-86233` (rel=2)
- `permed-75487` (rel=1)
- `permed-75490` (rel=1)
- `permed-73153` (rel=1)
- `permed-73149` (rel=1)

---

## Case 10: MedCorpus - Query topic_118_2

**Failure Type:** `ndcg_regression`
**Severity Score:** 0.176

### Query
> Considering these overall management challenges, how does optimizing preoperative risk factors specifically reduce postoperative complications in patients undergoing lower extremity bypass for CLTI?

### Profile
> Researcher in vascular surgery investigating preoperative risk factors and postoperative complications in patients undergoing lower extremity bypass for critical limb-threatening ischemia (CLTI).

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 1.00 | +0.00 |
| NDCG@10 | 0.933 | 0.758 | -0.176 |
| MRR@10 | 1.000 | 1.000 | +0.000 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `permed-11360` - 重视营养不良对慢性肢体威胁性缺血的影响...
2. [✓] `permed-85771` - Effectiveness of the Vascular Quality Initiative (VQI) Chronic Limb-Threatening ...
3. [✓] `permed-57080` - Characteristics, Antithrombotic Patterns, and Prognostic Outcomes in Claudicatio...
4. [✓] `permed-75575` - Factors Influencing Hospital Readmission after Lower Extremity Bypass for Chroni...
5. [✓] `permed-82519` - [Single-Center Experience of Treating Chronic Limb-Threatening Ischemia of Lower...

**Personalized (Profile+Query):**

1. [✓] `permed-75575` - Factors Influencing Hospital Readmission after Lower Extremity Bypass for Chroni...
2. [✓] `permed-85771` - Effectiveness of the Vascular Quality Initiative (VQI) Chronic Limb-Threatening ...
3. [✓] `permed-54815` - Incidence and Risk Factors of Postoperative Complications in General Surgery Pat...
4. [✓] `permed-57080` - Characteristics, Antithrombotic Patterns, and Prognostic Outcomes in Claudicatio...
5. [✓] `permed-11360` - 重视营养不良对慢性肢体威胁性缺血的影响...

### Ground Truth Relevant Documents

- `permed-82519` (rel=2)
- `permed-11360` (rel=2)
- `permed-75575` (rel=1)
- `permed-85771` (rel=1)
- `permed-90860` (rel=1)

---

## Case 11: MedCorpus - Query topic_042_3

**Failure Type:** `ndcg_regression`
**Severity Score:** 0.160

### Query
> In-depth analysis of ethical frameworks and socio-medical justifications underpinning specific highly divergent pediatric organ allocation policies identified, and their correlation with long-term patient survival and quality of life metrics

### Profile
> Researcher in medical ethics and pediatric transplantation investigating ethical frameworks and socio-medical justifications of specific highly divergent pediatric organ allocation policies and their correlation with long-term patient survival and quality of life metrics.

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 1.00 | +0.00 |
| NDCG@10 | 0.982 | 0.822 | -0.160 |
| MRR@10 | 1.000 | 1.000 | +0.000 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `permed-74645` - Allocation to pediatric recipients around the world: An IPTA global survey of cu...
2. [✓] `permed-52843` - International Pediatric Transplant Association (IPTA) position statement support...
3. [✓] `permed-87344` - Organ Dysfunction Among Children Meeting Brain Death Criteria: Implications for ...
4. [✓] `permed-74646` - Deceased donor organ allocation in pediatric transplantation: A historical narra...
5. [✓] `permed-74647` - Pediatric impacts of multiorgan transplant allocation policy in the United State...

**Personalized (Profile+Query):**

1. [✓] `permed-74647` - Pediatric impacts of multiorgan transplant allocation policy in the United State...
2. [✓] `permed-74645` - Allocation to pediatric recipients around the world: An IPTA global survey of cu...
3. [✓] `permed-67226` - Cultural and other beliefs as barriers to pediatric solid organ transplantation....
4. [✓] `permed-74646` - Deceased donor organ allocation in pediatric transplantation: A historical narra...
5. [✓] `permed-52843` - International Pediatric Transplant Association (IPTA) position statement support...

### Ground Truth Relevant Documents

- `permed-74646` (rel=2)
- `permed-52843` (rel=2)
- `permed-74645` (rel=2)
- `permed-74647` (rel=1)
- `permed-87344` (rel=1)

---

## Case 12: MedCorpus - Query topic_084_3

**Failure Type:** `ndcg_regression`
**Severity Score:** 0.128

### Query
> Technical approaches for adapting existing bioethical frameworks to systematically integrate underrepresented stakeholder perspectives, aiming for inclusive and practical guidelines in complex human-nonhuman interaction scenarios.

### Profile
> Researcher in bioethics exploring technical approaches for adapting existing bioethical frameworks to systematically integrate underrepresented stakeholder perspectives in complex human-nonhuman interaction scenarios, aiming for inclusive and practical guidelines.

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 1.00 | +0.00 |
| NDCG@10 | 0.949 | 0.821 | -0.128 |
| MRR@10 | 1.000 | 1.000 | +0.000 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `permed-54375` - Bioethics in the Current Climate....
2. [✓] `permed-72676` - A Call for Behavioral Science in Embedded Bioethics....
3. [✓] `permed-74865` - Why the World Needs Bioethics Communication....
4. [✓] `permed-72675` - Environmentalizing Bioethics: Planetary Health in a Perfect Moral Storm....
5. [✓] `permed-53038` - A Translational Role for Bioethics: Looking Back and Moving Forward....

**Personalized (Profile+Query):**

1. [✓] `permed-53038` - A Translational Role for Bioethics: Looking Back and Moving Forward....
2. [✓] `permed-54375` - Bioethics in the Current Climate....
3. [✓] `permed-72676` - A Call for Behavioral Science in Embedded Bioethics....
4. [✓] `permed-55968` - Finding the common ground: toward an inclusive approach to humanitarian crisis....
5. [✓] `permed-87900` - Why good work in philosophical bioethics often looks strange....

### Ground Truth Relevant Documents

- `permed-54375` (rel=2)
- `permed-72675` (rel=2)
- `permed-53038` (rel=1)
- `permed-72676` (rel=1)
- `permed-87900` (rel=1)

---

## Case 13: MedCorpus - Query topic_156_3

**Failure Type:** `ndcg_regression`
**Severity Score:** 0.118

### Query
> Legal and medical protocols for implementing pediatric euthanasia considering recent palliative care advancements and ethical frameworks

### Profile
> Researcher in medical ethics and pediatrics investigating legal and medical protocols for implementing pediatric euthanasia considering recent palliative care advancements and ethical frameworks.

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 1.00 | +0.00 |
| NDCG@10 | 1.000 | 0.882 | -0.118 |
| MRR@10 | 1.000 | 1.000 | +0.000 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `permed-74819` - Should the Dutch Law on Euthanasia Be Expanded to Include Children?...
2. [✓] `permed-80808` - The needs of children receiving end of life care and the impact of a paediatric ...
3. [✓] `permed-80197` - Honoring Long-Lived Cultural Beliefs for End-of-Life Care: Are We Prepared in th...
4. [✓] `permed-74821` - Medical Faculty Students' Views on Euthanasia: Does It Change With Medical Educa...
5. [✓] `permed-74823` - End-of-life care for patients with prolonged disorders of consciousness followin...

**Personalized (Profile+Query):**

1. [✓] `permed-80808` - The needs of children receiving end of life care and the impact of a paediatric ...
2. [✓] `permed-74819` - Should the Dutch Law on Euthanasia Be Expanded to Include Children?...
3. [✓] `permed-74821` - Medical Faculty Students' Views on Euthanasia: Does It Change With Medical Educa...
4. [✓] `permed-74711` - Ethics of Treatment Decisions for Extremely Premature Newborns with Poor Prognos...
5. [✓] `permed-90537` - [Ethical aspects of mechanical resuscitation in a child : Results of an expert w...

### Ground Truth Relevant Documents

- `permed-74819` (rel=2)
- `permed-80808` (rel=1)
- `permed-74711` (rel=1)
- `permed-76010` (rel=1)
- `permed-90537` (rel=1)

---

## Case 14: MedCorpus - Query topic_119_1

**Failure Type:** `ndcg_regression`
**Severity Score:** 0.105

### Query
> What methods are being researched to improve the biodegradation of waste-activated sludge?

### Profile
> Researcher in environmental engineering investigating biodegradation methods for waste-activated sludge in wastewater treatment processes.

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 1.00 | +0.00 |
| NDCG@10 | 0.981 | 0.877 | -0.105 |
| MRR@10 | 1.000 | 1.000 | +0.000 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `permed-79493` - Augmentation in polyhydroxybutyrate and biogas production from waste activated s...
2. [✓] `permed-79492` - Novel Free Nitrous Acid and Ultrasonication Pretreatment Enhanced Sludge Biodegr...
3. [✓] `permed-82401` - Moderate potassium ferrate dosage enhances methane production from the anaerobic...
4. [✓] `permed-79501` - Improved Dewaterability of Waste Activated Sludge by Fe(II)-Activated Potassium ...
5. [✓] `permed-79496` - Effect of activated carbon/graphite on enhancing anaerobic digestion of waste ac...

**Personalized (Profile+Query):**

1. [✓] `permed-79501` - Improved Dewaterability of Waste Activated Sludge by Fe(II)-Activated Potassium ...
2. [✓] `permed-79492` - Novel Free Nitrous Acid and Ultrasonication Pretreatment Enhanced Sludge Biodegr...
3. [✓] `permed-79493` - Augmentation in polyhydroxybutyrate and biogas production from waste activated s...
4. [✓] `permed-79496` - Effect of activated carbon/graphite on enhancing anaerobic digestion of waste ac...
5. [✓] `permed-66634` - Dissimilatory manganese reduction facilitates synergistic cooperation of hydroly...

### Ground Truth Relevant Documents

- `permed-79492` (rel=2)
- `permed-79496` (rel=2)
- `permed-82401` (rel=2)
- `permed-66634` (rel=2)
- `permed-79493` (rel=2)

---

## Case 15: MedCorpus - Query topic_054_3

**Failure Type:** `other`
**Severity Score:** 0.091

### Query
> Personalized medicine for NAFLD-associated liver fibrosis: tailoring treatment based on drug efficacy/safety profiles and patient stratification biomarkers

### Profile
> Researcher in hepatology investigating personalized medicine for NAFLD-associated liver fibrosis, focusing on tailoring treatment based on drug efficacy/safety profiles and patient stratification biomarkers in clinical treatment settings.

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 1.00 | +0.00 |
| NDCG@10 | 0.887 | 0.796 | -0.091 |
| MRR@10 | 1.000 | 1.000 | +0.000 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `permed-60269` - Noninvasive serum biomarkers for liver fibrosis in NAFLD: current and future....
2. [✓] `permed-52641` - Risk stratifying non-alcoholic fatty liver disease patients for optimal care Non...
3. [✓] `permed-52637` - Identification of high-risk subjects in NAFLD....
4. [✓] `permed-75520` - Side effect profile of pharmacologic therapies for liver fibrosis in nonalcoholi...
5. [✓] `permed-62718` - SWOT analysis of noninvasive tests for diagnosing NAFLD with severe fibrosis: an...

**Personalized (Profile+Query):**

1. [✓] `permed-84831` - Identified in blood diet-related methylation changes stratify liver biopsies of ...
2. [✓] `permed-60269` - Noninvasive serum biomarkers for liver fibrosis in NAFLD: current and future....
3. [✓] `permed-75520` - Side effect profile of pharmacologic therapies for liver fibrosis in nonalcoholi...
4. [✓] `permed-52637` - Identification of high-risk subjects in NAFLD....
5. [✓] `permed-52636` - Prognostication in NAFLD: physiological bases, clinical indicators, and newer bi...

### Ground Truth Relevant Documents

- `permed-52641` (rel=2)
- `permed-62693` (rel=1)
- `permed-52636` (rel=1)
- `permed-75520` (rel=1)
- `permed-52637` (rel=1)

---

## Case 16: LitSearch - Query 559

**Failure Type:** `both_miss`
**Severity Score:** 0.082

### Query
> What paper showed first that one can build a fully differentiable mixture of experts layer with no increase in time complexity?

### Profile
> Research on What paper showed first that one can build a fully differentiable mixture of experts layer with no increase in time complexity?

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 0.00 | 0.00 | +0.00 |
| NDCG@10 | 0.387 | 0.333 | -0.054 |
| MRR@10 | 0.200 | 0.143 | -0.057 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✗] `252668882` - SPARSE MIXTURE-OF-EXPERTS ARE DOMAIN GENER- ALIZABLE LEARNERS...
2. [✗] `247218249` - Parameter-Efficient Mixture-of-Experts Architecture for Pre-trained Language Mod...
3. [✗] `259342096` - Mixture-of-Experts Meets Instruction Tuning: A Winning Combination for Large Lan...
4. [✗] `261697072` - Pushing Mixture of Experts to the Limit: Extremely Parameter Efficient MoE for I...
5. [✓] `260378993` - From Sparse to Soft Mixtures of Experts...

**Personalized (Profile+Query):**

1. [✗] `12462234` - OUTRAGEOUSLY LARGE NEURAL NETWORKS: THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER...
2. [✗] `252668882` - SPARSE MIXTURE-OF-EXPERTS ARE DOMAIN GENER- ALIZABLE LEARNERS...
3. [✗] `261697072` - Pushing Mixture of Experts to the Limit: Extremely Parameter Efficient MoE for I...
4. [✗] `247958465` - MoEfication: Transformer Feed-forward Layers are Mixtures of Experts...
5. [✗] `11492613` - Learning Factored Representations in a Deep Mixture of Experts...

### Ground Truth Relevant Documents

- `260378993` (rel=1)

---

## Case 17: MedCorpus - Query topic_084_2

**Failure Type:** `other`
**Severity Score:** 0.077

### Query
> Methods for enhancing this bioethical role to better incorporate diverse perspectives and community values in human-nonhuman contexts

### Profile
> Researcher in bioethics exploring methods to enhance bioethical role by integrating diverse perspectives and community values within human-nonhuman contexts, emphasizing interdisciplinary ethical frameworks and stakeholder engagement.

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 1.00 | +0.00 |
| NDCG@10 | 0.912 | 0.835 | -0.077 |
| MRR@10 | 1.000 | 1.000 | +0.000 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `permed-54375` - Bioethics in the Current Climate....
2. [✓] `permed-53039` - Running Toward Disasters: One Bioethicist's Experience in Translational Ethics....
3. [✓] `permed-53038` - A Translational Role for Bioethics: Looking Back and Moving Forward....
4. [✓] `permed-53040` - A Bioethics for Democracy: Restoring Civic Vision....
5. [✓] `permed-72676` - A Call for Behavioral Science in Embedded Bioethics....

**Personalized (Profile+Query):**

1. [✓] `permed-53038` - A Translational Role for Bioethics: Looking Back and Moving Forward....
2. [✓] `permed-54375` - Bioethics in the Current Climate....
3. [✓] `permed-87900` - Why good work in philosophical bioethics often looks strange....
4. [✓] `permed-53039` - Running Toward Disasters: One Bioethicist's Experience in Translational Ethics....
5. [✓] `permed-53040` - A Bioethics for Democracy: Restoring Civic Vision....

### Ground Truth Relevant Documents

- `permed-54375` (rel=2)
- `permed-69886` (rel=2)
- `permed-53038` (rel=1)
- `permed-87900` (rel=1)
- `permed-53040` (rel=1)

---

## Case 18: MedCorpus - Query topic_161_2

**Failure Type:** `other`
**Severity Score:** 0.077

### Query
> Can modifying the ligand environment in these complexes enhance their reactivity with certain compounds?

### Profile
> Research on Can modifying the ligand environment in these complexes enhance their reactivity with certain compounds?

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 1.00 | +0.00 |
| NDCG@10 | 0.973 | 0.896 | -0.077 |
| MRR@10 | 1.000 | 1.000 | +0.000 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `permed-52518` - Diquinol Functionality Boosts the Superoxide Dismutase Mimicry of a Zn(II) Compl...
2. [✓] `permed-84013` - Properties of Amine-Containing Ligands That Are Necessary for Visible-Light-Prom...
3. [✓] `permed-72933` - Direct arylation reaction catalyzed by a PEPPSI-type palladium complex with an a...
4. [✓] `permed-78876` - Octahedral Rhenium Cluster Complexes with 1,2-Bis(4-pyridyl)ethylene and 1,3-Bis...
5. [✓] `permed-78859` - Diantimony Complexes [(CpR)2Mo2(CO)4(μ,η2-Sb2)] (CpR = C5H5, C5H4tBu) as Unexpec...

**Personalized (Profile+Query):**

1. [✓] `permed-52518` - Diquinol Functionality Boosts the Superoxide Dismutase Mimicry of a Zn(II) Compl...
2. [✓] `permed-78876` - Octahedral Rhenium Cluster Complexes with 1,2-Bis(4-pyridyl)ethylene and 1,3-Bis...
3. [✓] `permed-78859` - Diantimony Complexes [(CpR)2Mo2(CO)4(μ,η2-Sb2)] (CpR = C5H5, C5H4tBu) as Unexpec...
4. [✓] `permed-57981` - How general is the effect of the bulkiness of organic ligands on the basicity of...
5. [✓] `permed-72933` - Direct arylation reaction catalyzed by a PEPPSI-type palladium complex with an a...

### Ground Truth Relevant Documents

- `permed-52518` (rel=2)
- `permed-72935` (rel=2)
- `permed-84013` (rel=2)
- `permed-72933` (rel=2)
- `permed-78859` (rel=1)

---

## Case 19: MedCorpus - Query topic_200_1

**Failure Type:** `other`
**Severity Score:** 0.065

### Query
> Healthcare access barriers for women who inject drugs in urban environments

### Profile
> Researcher in public health investigating healthcare access barriers for women who inject drugs in urban environments for harm reduction policies and social equity interventions.

### Metrics Comparison

| Metric | Baseline | Personalized | Delta |
|--------|----------|--------------|-------|
| P@1 | 1.00 | 1.00 | +0.00 |
| NDCG@10 | 0.984 | 0.920 | -0.065 |
| MRR@10 | 1.000 | 1.000 | +0.000 |

### Top-5 Documents Comparison

**Baseline (Query-only):**

1. [✓] `permed-83064` - Participant perceptions on the acceptability and feasibility of a telemedicine-b...
2. [✓] `permed-70149` - We want everything in a one-stop shop: acceptability and feasibility of PrEP and...
3. [✓] `permed-71368` - Barriers to PrEP uptake among Black female adolescents and emerging adults....
4. [✓] `permed-65780` - Engaging Latino sexual minority men in PrEP and behavioral health care: multilev...
5. [✓] `permed-65938` - Examining pharmacies' ability to increase pre-exposure prophylaxis access for bl...

**Personalized (Profile+Query):**

1. [✓] `permed-70149` - We want everything in a one-stop shop: acceptability and feasibility of PrEP and...
2. [✗] `permed-91926` - e5801...
3. [✓] `permed-70154` - Gender Differences in HIV, HCV risk and Prevention Needs Among People who Inject...
4. [✓] `permed-71368` - Barriers to PrEP uptake among Black female adolescents and emerging adults....
5. [✓] `permed-83064` - Participant perceptions on the acceptability and feasibility of a telemedicine-b...

### Ground Truth Relevant Documents

- `permed-70149` (rel=1)
- `permed-70154` (rel=1)
- `permed-63473` (rel=1)
- `permed-83064` (rel=1)
- `permed-71368` (rel=1)

---

---

# 失败案例分析报告（中文版）

**发现的失败案例总数：** 19  
**按严重程度排序，展示全部案例**

## 失败类型统计

| 失败类型 | 数量 | 说明 |
|---------|------|------|
| **top1_regression** | 8 | 个性化把原本正确的第1名文档推到后面 |
| **ndcg_regression** | 6 | NDCG@10 明显下降（>0.1） |
| **other** | 4 | 其他类型的轻微下降 |
| **both_miss** | 1 | 两种方法都没命中第1名，但个性化更差 |

---

## 典型失败案例详解

### 案例1：LitSearch Query 105（最严重）

**失败类型：** `top1_regression`（第1名回退）  
**严重程度：** 2.944

**查询：**
> 有没有研究将外部常识知识融入对话模型以增强情感支持？

**Profile描述：**
> 人工智能领域研究，关注将外部常识知识融入对话模型，通过改进上下文理解和共情响应来增强情感支持。

**指标对比：**

| 指标 | Baseline | 个性化 | 差异 |
|------|----------|--------|------|
| P@1 | 1.00 | 0.00 | -1.00 |
| NDCG@10 | 1.000 | 0.431 | -0.569 |
| MRR@10 | 1.000 | 0.250 | -0.750 |

**问题分析：**
- Baseline 正确将 `247748640`（MISC: 情感支持对话的混合策略模型）排在第1位
- 个性化后，该文档从第1位掉到第4位
- 第1位变成了一篇中文论文（基於常識知識的移情對話回覆生成），虽然主题相关但不是 ground truth

**根因：** Profile 过于宽泛，没有提供比原始查询更精确的信息，反而引入了干扰。

---

### 案例2：LitSearch Query 251

**失败类型：** `top1_regression`  
**严重程度：** 2.833

**查询：**
> 能推荐一些用对比学习方法开发多模态模型的工作吗？

**Profile描述：**
> Research on Could you suggest some work that develops multimodal models with contrastive learning approaches?

**问题分析：**
- Profile 直接复制了查询文本，没有提供任何额外信息
- 这种"无效 profile"反而干扰了检索，把正确答案从第1位推到第3位

**根因：** Profile 生成失败，变成了查询的简单重复。

---

### 案例3：MedCorpus topic_095_1

**失败类型：** `top1_regression`  
**严重程度：** 2.672

**查询：**
> SARS-CoV-2 感染是否有影响骨骼健康的已知并发症？

**Profile描述：**
> 传染病和肌肉骨骼健康研究者，调查SARS-CoV-2感染对骨骼健康的并发症影响。

**问题分析：**
- Baseline 正确将骨骼相关文献排在前面
- 个性化后，第1位变成了无关的 `permed-68513`（仅标题为"SARS-CoV-2"），第2位是心律失常文献
- 骨坏死相关的正确文献被推到第3-4位

**根因：** Profile 没有足够强调"骨骼健康"这个关键点，导致检索偏向更泛化的COVID-19文献。

---

### 案例16：LitSearch Query 559（稀缺证据案例）

**失败类型：** `both_miss`  
**严重程度：** 0.082

**查询：**
> 哪篇论文首次证明可以构建一个完全可微的专家混合层，且不增加时间复杂度？

**问题分析：**
- 这是一个寻找"首创性论文"的精确检索任务
- Baseline 和个性化都没能把正确答案 `260378993`（From Sparse to Soft Mixtures of Experts）排在第1位
- Baseline 排在第5位，个性化后甚至掉出了 Top-5
- 个性化把早期的 MoE 论文（2017年的 Shazeer 论文）排到第1位，时间线错误

**根因：** 这类"首创性"查询需要精确的时间线知识，profile 无法提供帮助。

---

## 失败原因分类总结

| 原因类别 | 案例数 | 典型特征 |
|---------|--------|----------|
| **Profile 内容空泛** | 6 | Profile 直接复制查询或过于泛化 |
| **领域焦点偏移** | 5 | Profile 引入了偏离核心主题的概念 |
| **精确检索任务** | 3 | 需要时间线/首创性等精确信息 |
| **多轮对话上下文丢失** | 3 | 第2-3轮查询中，profile 没有捕捉前文重点 |
| **标注噪声** | 2 | Ground truth 本身可能存在问题 |

---

## 改进建议

1. **Profile 质量检测**：在使用前检测 profile 是否为查询的简单重复
2. **动态权重**：对于精确检索任务（如"首次提出"），降低 profile 的权重
3. **多轮对话**：更好地保留前序轮次的关键实体和约束
4. **回退机制**：当个性化结果置信度低时，回退到 baseline

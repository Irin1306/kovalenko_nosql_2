# ArXiv Semantic Search — Pinecone

## Опис проєкту

Мета проєкту — навчитися працювати з векторною базою даних Pinecone, створювати embeddings для наукових статей, виконувати семантичний пошук та досліджувати вплив метрик схожості й стратегій розбиття тексту на якість пошуку.

**Технології:** Python 3, Pandas, PyTorch, HuggingFace Transformers, SPECTER2, Pinecone, NumPy, arXiv Dataset. 

### Завдання 1.1 Підготовка датасету

Результати виконання `01_prepare_data.py`: 
 
Завантажено 5000 статей.
Найбільша категорія — astro-ph (972 статті).
Усі статті належать до 2007 року.
Дані збережено у data/arxiv_subset.parquet.

Приклад запису:
```python
{
    'id': '0704.0001',
    'title': 'Calculation of prompt diphoton production cross sections...',
    'year': 2007,
    'category': 'hep-ph'
}
``` 

### Завдання 1.2 Вибір інструментів

1. Чим Pinecone відрізняється від Qdrant і Chroma за моделлю розгортання, ліцензією і продуктивністю? У якому сценарії ви б обрали кожен із них?

Pinecone — керована хмарна векторна база даних. Не потрібно самостійно налаштовувати сервер або підтримувати інфраструктуру. Це зручний варіант для швидкого запуску та масштабування пошукових систем.

Qdrant — векторна база даних з відкритим вихідним кодом під ліцензією Apache 2.0. Її можна запускати локально, через Docker або у хмарі. Підходить для проєктів, де потрібен контроль над інфраструктурою.

Chroma — відкрита векторна база даних під ліцензією Apache 2.0. Вона проста у встановленні та добре підходить для локальних експериментів і невеликих проєктів.

Pinecone підходить для керованого хмарного рішення, Qdrant — для гнучкого контролю над векторним пошуком, а Chroma — для швидкої розробки та локальних експериментів. 

2. Чому для задачі пошуку по науковим текстам обрана модель specter2_base, а не універсальна all-MiniLM-L6-v2? Знайдіть картку моделі на HuggingFace і процитуйте, для яких задач вона навчена.

SPECTER2 спеціально розроблена для роботи з науковими документами. Вона враховує зміст наукових статей, їхні теми та семантичну близькість. 
На відміну від неї, all-MiniLM-L6-v2 є універсальною моделлю для створення sentence embeddings.  

У проєкті використовується базова модель allenai/specter2_base разом із двома адаптерами: 
- allenai/specter2 — для створення embeddings наукових статей (Proximity / Retrieval).
- allenai/specter2_adhoc_query — для створення embeddings коротких пошукових запитів (Adhoc Search). 

У картці моделі HuggingFace зазначені такі типи задач для адаптерів SPECTER2: 
- Classification
- Regression
- Proximity (Retrieval)
- Adhoc Search

Посилання: https://huggingface.co/allenai/specter2_base

3. Що написано у картці моделі про рекомендовану метрику схожості? Чому це важливо при створенні індексу?

У картці моделі SPECTER2 на HuggingFace описано призначення моделі та чотири формати задач, для яких навчені адаптери: Classification, Regression, Proximity (Retrieval) та Adhoc Search. Водночас у картці базової моделі не зазначено обов'язкову або рекомендовану метрику схожості.

У проєкті використовується cosine similarity як обрана метрика пошуку в Pinecone

```python
pc.create_index(
    name=INDEX_NAME,
    dimension=VECTOR_DIM,
    metric="cosine",
    spec=ServerlessSpec(
        cloud=CLOUD,
        region=REGION
    )
)
```

### Завдання 1.3 Отримання ембеддингів  

1. Чому при використанні нормалізованих ембеддингів (одиничної довжини) косинусна схожість (cosine similarity) еквівалентна скалярному добутку (dot product)?
 
У проєкті ембеддинги нормалізуються за допомогою `torch.nn.functional.normalize()`:

**У файлі `02_embed.py`:**

```python
embeddings = torch.nn.functional.normalize(
    embeddings,
    p=2,
    dim=1
)
```   

Формула косинусної схожості:

```text
cos(A, B) = (A · B) / (||A|| * ||B||)
```

Для одиничних векторів:

```text
||A|| = 1
||B|| = 1
```

Тому:

```text
cos(A, B) = A · B
```

Отже, cosine similarity та dot product дають однаковий результат для нормалізованих векторів. 


### Завдання 2. Завантаження даних і метадані 

Результат виконання `03_load_to_pinecone.py`:

```text
Dataset records: 5000
Embeddings shape: (5000, 768)

Creating index: arxiv-papers
Index created.

Uploading vectors to Pinecone...
Uploading batches: 100%|██████████| 25/25

Upload completed!
Total vectors in index: 5000
```


### Завдання 3. Пошукові запити

Для пошуку використовується модель SPECTER2 з двома адаптерами:

- specter2_adhoc_query — для пошукових запитів.
- specter2 — для документів.

1. Чистий семантичний пошук

Запит:  

```python
query = "galaxy evolution star formation properties of galaxies"
```

Отримано топ-5 статей:

```text
Rank: 1
ID: paper_3610
Score: 0.8132
Title: UV Star Formation Rates in the Local Universe
Category: astro-ph
Year: 2007.0
Abstract: We measure star formation rates of ~50,000 optically-selected galaxies in the
local universe (z~0.1), spanning a range from gas-rich dwarfs to massive
ellipticals. We obtain dust-corrected SFRs by fitting the GALEX (UV) and SDSS
(optical) photometry 

Rank: 2
ID: paper_2704
Score: 0.7994
Title: Chemical Properties of Star Forming Dwarf Galaxies
Category: astro-ph
Year: 2007.0
Abstract: Recent studies of the near-infrared (NIR) properties of dwarf irregular
galaxies (dIs) and blue compact dwarfs (BCDs) have provided improved estimates
for the NIR luminosity of old stellar populations in these galaxies. These can
be used to derive ga

Rank: 3
ID: paper_3076
Score: 0.7933
Title: Star Formation and the Growth of Stellar Mass
Category: astro-ph
Year: 2007.0
Abstract: Recent observations have demonstrated a significant growth in the integrated
stellar mass of the red sequence since z=1, dominated by a steadily increasing
number of galaxies with stellar masses M* < 10^11 M_sun. In this paper, we use
the COMBO-17 ph

Rank: 4
ID: paper_1759
Score: 0.7932
Title: Galaxy Evolution and Environment
Category: astro-ph
Year: 2007.0
Abstract: The properties of galaxies are strongly correlated with their environment,
with red galaxies dominating galaxy clusters and blue galaxies dominating the
general field. However, not all field galaxies are young: studies of the
colors, line strengths, 

Rank: 5
ID: paper_1482
Score: 0.7912
Title: Exploring Infrared Properties of Giant Low Surface Brightness Galaxies
Category: astro-ph
Year: 2007.0
Abstract: Abridged: We present analysis of Spitzer Space Telescope observations of the
three low surface brightness (LSB) optical giant galaxies Malin 1, UGC 6614 and
UGC 9024. Mid- and far-infrared morphology, spectral energy distributions, and
integrated col
``` 

2. Пошук із фільтрацією

Завдання A. Статті з 2021 року, категорія astro-ph.

Результат: No results found.

У вибірці не знайдено документів, які відповідають умовам фільтрації.

Завдання B. Статті до 2015 року.

Отримано п'ять статей 2007 року, які збігаються з результатами чистого семантичного пошуку. 

3. Порівняння метрик схожості на локальних ембеддингах 

```python
embeddings = np.load(EMBEDDINGS_FILE)
Розмірність: (5000, 768).
``` 

Обчислено три метрики:

- Cosine Similarity — більше значення означає більшу схожість.
- Dot Product — більше значення означає більшу схожість.
- L2 Distance — менше значення означає більшу схожість.

Для кожної метрики було отримано топ-5 статей.

Результати експерименту: 
Метрика	Топ-5 спільних результатів
Cosine vs Dot Product	5/5
Cosine vs L2	5/5
Dot Product vs L2	5/5

Усі три метрики повернули однакові п'ять статей у тому самому порядку. 
Це пояснюється тим, що ембеддинги документів і пошуковий вектор нормалізовані.


### Завдання 4. Chunking

Результати виконання `05_chunking.py`: 

```text
ARXIV-CHUNKS-FIXED
======================================================================

Validating chunk token lengths...
Maximum chunk length: 234 tokens
Average chunk length: 175.3 tokens

Generating embeddings...
Embedding batches: 100%|████████████████████████████████| 6/6 [00:17<00:00,  2.94s/it]
Embeddings shape: (90, 768)
First vector norm: 1.000000

Deleting old index: arxiv-chunks-fixed

Creating index: arxiv-chunks-fixed
Index arxiv-chunks-fixed created successfully.

Uploading chunks to Pinecone...
Uploading: 100%|████████████████████████████████████████| 1/1 [00:01<00:00,  1.85s/it]

Upload completed.
Total vectors in index: 90

======================================================================
ARXIV-CHUNKS-SEMANTIC
======================================================================

Validating chunk token lengths...
Maximum chunk length: 227 tokens
Average chunk length: 153.1 tokens

Generating embeddings...
Embedding batches: 100%|████████████████████████████████| 6/6 [00:18<00:00,  3.06s/it]
Embeddings shape: (93, 768)
First vector norm: 1.000000

Deleting old index: arxiv-chunks-semantic

Creating index: arxiv-chunks-semantic
Index arxiv-chunks-semantic created successfully.

Uploading chunks to Pinecone...
Uploading: 100%|████████████████████████████████████████| 1/1 [00:01<00:00,  1.70s/it]

Upload completed.
Total vectors in index: 93
```

Для перевірки використовувалися три запити: 

1. Запит 1 —"How is dark matter distributed in the Milky Way?"

```text
----------------------------------------------------------------------
FIXED-SIZE SEARCH
----------------------------------------------------------------------

Rank: 1
Score: 0.9168
Article: Dark matter in the Milky Way, II. the HI gas distribution as a tracer of
  the gravitational potential
arXiv ID: 0704.3925
Year: 2007.0
Category: astro-ph
Chunk number: 0.0
Text: Context. Gas within a galaxy is forced to establish pressure balance against gravitational forces. The shape of an unperturbed gaseous disk can be used to constrain dark matter models. Aims. We derive the 3-D HI volume density distribution for the Milky Way out to a galactocentric radius of 40 kpc and a height of 20 kpc to constrain the Galactic ma

Rank: 2
Score: 0.9126
Article: Dark matter in the Milky Way, II. the HI gas distribution as a tracer of
  the gravitational potential
arXiv ID: 0704.3925
Year: 2007.0
Category: astro-ph
Chunk number: 1.0
Text: to solve the combined Poisson-Boltzmann equation in a self-consistent way and optimized to reproduce the observed flaring. Results. Besides a massive extended halo of M ~ 1.8 10^{12} Msun, we find a self-gravitating dark matter disk with M=2 to 3 10^{11} Msun, including a dark matter ring at 13 < R < 18.5 kpc with M = 2.2 to 2.8 10^{10} Msun. The e

Rank: 3
Score: 0.9078
Article: The Evolution of Dark Matter Halo Properties in Clusters, Filaments,
  Sheets and Voids
arXiv ID: 0704.2595
Year: 2007.0
Category: astro-ph
Chunk number: 1.0
Text: with filaments and sheets. In detail, at all redshifts up to z=1: a) Haloes with M<M* tend to be more oblate when located in clusters than in the other environments;this trend is reversed at higher masses: above about M*, halos in clusters are typically more prolate than similar massive haloes in sheets, filaments and voids. b) Haloeswith M>M* in

Rank: 4
Score: 0.9055
Article: Dark matter in the Milky Way, II. the HI gas distribution as a tracer of
  the gravitational potential
arXiv ID: 0704.3925
Year: 2007.0
Category: astro-ph
Chunk number: 2.0
Text: The \hi gas layer is strongly flaring. The HWHM scale height is 60 pc at R = 4 kpc and increases to ~2700$ pc at R=40 kpc. Spiral arms cause a noticeable imprint on the gravitational field, at least out to R = 30 kpc. Conclusions. Our mass model supports previous proposals that the giant stellar ring structure is due to a merging dwarfgalaxy. The 

Rank: 5
Score: 0.9041
Article: Identifying Dark Matter Burners in the Galactic center
arXiv ID: 0704.1324
Year: 2007.0
Category: astro-ph
Chunk number: 2.0
Text: be consistent with WIMP burners in the form of degenerate cores with envelopes.We also cross-check the WIMP burner hypothesis with the EGRET observed flux of gamma-rays from the Galactic center, which imposes a constraint on the dark matter spike density profile and annihilation cross-section. We find that the EGRET data is consistentwith the WIM

----------------------------------------------------------------------
SEMANTIC SEARCH
----------------------------------------------------------------------

Rank: 1
Score: 0.9198
Article: The Evolution of Dark Matter Halo Properties in Clusters, Filaments,
  Sheets and Voids
arXiv ID: 0704.2595
Year: 2007.0
Category: astro-ph
Chunk number: 2.0
Text: haloes with M>M* perpendicular to it.

Rank: 2
Score: 0.9140
Article: Dark matter in the Milky Way, II. the HI gas distribution as a tracer of
  the gravitational potential
arXiv ID: 0704.3925
Year: 2007.0
Category: astro-ph
Chunk number: 0.0
Text: Context. Gas within a galaxy is forced to establish pressure balance against gravitational forces. The shape of an unperturbed gaseous disk can be used to constrain dark matter models. Aims. We derive the 3-D HI volume density distribution for the Milky Way out to a galactocentric radius of 40 kpc and a height of 20 kpc to constrain the Galactic ma

Rank: 3
Score: 0.9114
Article: The Evolution of Dark Matter Halo Properties in Clusters, Filaments,
  Sheets and Voids
arXiv ID: 0704.2595
Year: 2007.0
Category: astro-ph
Chunk number: 1.0
Text: In detail, at all redshifts up to z=1: a) Haloes with M<M* tend to be more oblate when located in clusters than in the other environments; this trend is reversed at higher masses: above about M*, halos in clusters are typically more prolate than similar massive haloes in sheets, filaments and voids. b) Haloes with M>M* in filaments spin more rapidl

Rank: 4
Score: 0.9096
Article: Identifying Dark Matter Burners in the Galactic center
arXiv ID: 0704.1324
Year: 2007.0
Category: astro-ph
Chunk number: 2.0
Text: We also cross-check the WIMP burner hypothesis with the EGRET observed flux of gamma-rays from the Galactic center, which imposes a constraint on the dark matter spike density profile and annihilation cross-section. We find that the EGRET data is consistent with the WIMP burner hypothesis. New high precision measurements by GLAST will confirm or se

Rank: 5
Score: 0.9094
Article: Dark matter in the Milky Way, II. the HI gas distribution as a tracer of
  the gravitational potential
arXiv ID: 0704.3925
Year: 2007.0
Category: astro-ph
Chunk number: 1.0
Text: Besides a massive extended halo of M ~ 1.8 10^{12} Msun, we find a self-gravitating dark matter disk with M=2 to 3 10^{11} Msun, including a dark matter ring at 13 <R < 18.5 kpc with M = 2.2 to 2.8 10^{10} Msun. The existence of the ring was previously postulated from EGRET data and coincides with a giant stellar structure that surrounds the Galax
``` 

2. Запит 2 — "What are the methods for detecting extrasolar planets?"

```text
----------------------------------------------------------------------
FIXED-SIZE SEARCH
----------------------------------------------------------------------

Rank: 1
Score: 0.9134
Article: An Imaging Survey for Extrasolar Planets around 45 Close, Young Stars
  with SDI at the VLT and MMT
arXiv ID: 0705.0066
Year: 2007.0
Category: astro-ph
Chunk number: 2.0
Text: at 2 AU from a 12 Myr M star at 10 pc. We believe that our SDI images are the highest contrast astronomical images ever made from ground or space for methane rich companions <1'' from their star. For the best 20 of our survey stars, we attained 50% 5 sigma completeness for 6-10 M_Jup planets at semi-major axes of 20-40 AU. Thus, our completeness le

Rank: 2
Score: 0.9127
Article: IRAS 18317-0757: A Cluster of Embedded Massive Stars and Protostars
arXiv ID: 0704.1301
Year: 2007.0
Category: astro-ph
Chunk number: 1.0
Text: the other four objects, the nondetections at 2.7 mm and in the MSX mid-IR bandsare consistent with cool dust emission with a temperature of 13-40K and luminosity of1000-40000 Lsun. By combining single-dish and interferometric data, we have identified over two dozen virialized C18O cores in this region that contain ~40% of the total molecular gas 

Rank: 3
Score: 0.9124
Article: An Imaging Survey for Extrasolar Planets around 45 Close, Young Stars
  with SDI at the VLT and MMT
arXiv ID: 0705.0066
Year: 2007.0
Category: astro-ph
Chunk number: 0.0
Text: We present the results of a survey of 45 young (<250 Myr), close (<50 pc) starswith the Simultaneous Differential Imager (SDI) implemented at the VLT and the MMT for the direct detection of extrasolar planets. As part of the survey, we observed 54 objects total, of which 45 were close, young stars, 2 were somewhat more distant (<150 pc), very youn

Rank: 4
Score: 0.9018
Article: SW Sextantis stars: the dominant population of CVs with orbital periods
  between 3-4 hours
arXiv ID: 0704.1129
Year: 2007.0
Category: astro-ph
Chunk number: 2.0
Text: These statistics are confirmed by our results from the Hamburg Quasar Survey CVs. Remarkably, 54 per cent of the Hamburg nova-like variables have been identified as SW Sex stars with orbital periods in the 3-4.5 h range. The observation of this pile-up of systems close to the upper boundary of the period gap is difficult to reconcile with the stand

Rank: 5
Score: 0.8996
Article: Observations towards early-type stars in the ESO-POP survey: II --
  searches for intermediate and high velocity clouds
arXiv ID: 0704.1315
Year: 2007.0
Category: astro-ph
Chunk number: 1.0
Text: at +122 km/s and NHI=1.2x10^19 cm-2 is seen. The CaII K spectrum has a S/N = 780, although no absorption is observed at the cloud velocity. This similarly places a firm lower distance limit of 2.9-kpc towards this parcel of gas that may be an intermediate velocity cloud. The lack of intermediate velocity (IV) CaII K absorption towards HD 196426 at 

----------------------------------------------------------------------
SEMANTIC SEARCH
----------------------------------------------------------------------

Rank: 1
Score: 0.9272
Article: An Imaging Survey for Extrasolar Planets around 45 Close, Young Stars
  with SDI at the VLT and MMT
arXiv ID: 0705.0066
Year: 2007.0
Category: astro-ph
Chunk number: 3.0
Text: From our survey null result, we can rule out (at the 98% confidence/2.0sigma level) a model planet population using a planet distribution where N(a) $\propto$ constant out to a distance of 45 AU (further model assumptions discussed within).

Rank: 2
Score: 0.9188
Article: An Imaging Survey for Extrasolar Planets around 45 Close, Young Stars
  with SDI at the VLT and MMT
arXiv ID: 0705.0066
Year: 2007.0
Category: astro-ph
Chunk number: 0.0
Text: We present the results of a survey of 45 young (<250 Myr), close (<50 pc) starswith the Simultaneous Differential Imager (SDI) implemented at the VLT and the MMT for the direct detection of extrasolar planets. As part of the survey, we observed 54 objects total, of which 45 were close, young stars, 2 were somewhat more distant (<150 pc), very youn

Rank: 3
Score: 0.9157
Article: SW Sextantis stars: the dominant population of CVs with orbital periods
  between 3-4 hours
arXiv ID: 0704.1129
Year: 2007.0
Category: astro-ph
Chunk number: 2.0
Text: Remarkably, 54 per cent of the Hamburg nova-like variables have been identifiedas SW Sex stars with orbital periods in the 3-4.5 h range. The observation of this pile-up of systems close to the upper boundary of the period gap is difficult to reconcile with the standard theory of CV evolution, as the SW Sex stars are believed to have the highest m

Rank: 4
Score: 0.9104
Article: IRAS 18317-0757: A Cluster of Embedded Massive Stars and Protostars
arXiv ID: 0704.1301
Year: 2007.0
Category: astro-ph
Chunk number: 1.0
Text: For the other four objects, the nondetections at 2.7 mm and in the MSX mid-IR bands are consistent with cool dust emission with a temperature of 13-40K and luminosity of 1000-40000 Lsun. By combining single-dish and interferometric data, we have identified over two dozen virialized C18O cores in this region that contain ~40% of the total molecular 

Rank: 5
Score: 0.9035
Article: An Imaging Survey for Extrasolar Planets around 45 Close, Young Stars
  with SDI at the VLT and MMT
arXiv ID: 0705.0066
Year: 2007.0
Category: astro-ph
Chunk number: 2.0
Text: With this degree of attenuation, we should be able to image (5sigma detection) a 5 M_{Jup} planet 15 AU from a 70 Myr K1 star at 15 pc or a 5 M_{Jup} planet at 2 AU from a 12 Myr M star at 10 pc. We believe that our SDI images are the highest contrastastronomical images ever made from ground or space for methane rich companions <1'' from their sta
``` 

3. Запит 3 — "What is the relationship between active galactic nuclei and black hole accretion?"

```text
----------------------------------------------------------------------
FIXED-SIZE SEARCH
----------------------------------------------------------------------

Rank: 1
Score: 0.9151
Article: On the Origin of the Dichotomy of Early-Type Galaxies: The Role of Dry
  Mergers and AGN Feedback
arXiv ID: 0704.0932
Year: 2007.0
Category: astro-ph
Chunk number: 2.0
Text: lower, so that their last major merger was dryer. Finally, our model predicts that (i) less than 10 percent of all early-type galaxies form in major mergers that involve two early-type progenitors, (ii) more than 95 percent of all boxy early-type galaxies with $M_* < 2 \times 10^{10} h^{-1} \Msun$ are satellite galaxies, and (iii) about 70 percent 

Rank: 2
Score: 0.9109
Article: Identifying Dark Matter Burners in the Galactic center
arXiv ID: 0704.1324
Year: 2007.0
Category: astro-ph
Chunk number: 1.0
Text: vicinity of an adiabatically grown SMBH. We find that the most efficient WIMP burners are stars with degenerate electron cores, e.g. white dwarfs (WD) or degenerate cores with envelopes. If found, such stars would provide evidence for the existence ofparticle dark matter and could possibly be used to establish its density profile. In our previous 

Rank: 3
Score: 0.9106
Article: Identifying Dark Matter Burners in the Galactic center
arXiv ID: 0704.1324
Year: 2007.0
Category: astro-ph
Chunk number: 0.0
Text: If the supermassive black hole (SMBH) at the center of our Galaxy grew adiabatically, then a dense "spike" of dark matter is expected to have formed around it. Assuming that dark matter is composed primarily of weakly interacting massive particles (WIMPs), a star orbiting close enough to the SMBH can capture WIMPs at an extremely high rate. The ste

Rank: 4
Score: 0.9019
Article: An absorption origin for the soft excess in Seyfert 1 AGN
arXiv ID: 0704.2970
Year: 2007.0
Category: astro-ph
Chunk number: 2.0
Text: spectral state in X-ray binaries. This does not always show that the disc in NLS1's contributes less than 80 per cent of the total power, as expected for a very highstate. We suggest that this is an artifact of the standard disc models used to determine the disc luminosity in our fits. The disc seen in the very high state of black hole binaries is

Rank: 5
Score: 0.9008
Article: Dark matter in the Milky Way, II. the HI gas distribution as a tracer of
  the gravitational potential
arXiv ID: 0704.3925
Year: 2007.0
Category: astro-ph
Chunk number: 2.0
Text: The \hi gas layer is strongly flaring. The HWHM scale height is 60 pc at R = 4 kpc and increases to ~2700$ pc at R=40 kpc. Spiral arms cause a noticeable imprint on the gravitational field, at least out to R = 30 kpc. Conclusions. Our mass model supports previous proposals that the giant stellar ring structure is due to a merging dwarfgalaxy. The 

----------------------------------------------------------------------
SEMANTIC SEARCH
----------------------------------------------------------------------

Rank: 1
Score: 0.9132
Article: The Evolution of Dark Matter Halo Properties in Clusters, Filaments,
  Sheets and Voids
arXiv ID: 0704.2595
Year: 2007.0
Category: astro-ph
Chunk number: 2.0
Text: haloes with M>M* perpendicular to it.

Rank: 2
Score: 0.9125
Article: On the Origin of the Dichotomy of Early-Type Galaxies: The Role of Dry
  Mergers and AGN Feedback
arXiv ID: 0704.0932
Year: 2007.0
Category: astro-ph
Chunk number: 2.0
Text: Finally, our model predicts that (i) less than 10 percent of all early-type galaxies form in major mergers that involve two early-type progenitors, (ii) more than 95percent of all boxy early-type galaxies with $M_* < 2 \times 10^{10} h^{-1} \Msun$ are satellite galaxies, and (iii) about 70 percent of all low mass early-types do not form a supermas

Rank: 3
Score: 0.9104
Article: Identifying Dark Matter Burners in the Galactic center
arXiv ID: 0704.1324
Year: 2007.0
Category: astro-ph
Chunk number: 0.0
Text: If the supermassive black hole (SMBH) at the center of our Galaxy grew adiabatically, then a dense "spike" of dark matter is expected to have formed around it. Assuming that dark matter is composed primarily of weakly interacting massive particles (WIMPs), a star orbiting close enough to the SMBH can capture WIMPs at an extremely high rate. The ste

Rank: 4
Score: 0.9036
Article: An absorption origin for the soft excess in Seyfert 1 AGN
arXiv ID: 0704.2970
Year: 2007.0
Category: astro-ph
Chunk number: 2.0
Text: This does not always show that the disc in NLS1's contributes less than 80 per cent of the total power, as expected for a very high state. We suggest that this is anartifact of the standard disc models used to determine the disc luminosity in our fits. The disc seen in the very high state of black hole binaries is often observed to bedistorted fr

Rank: 5
Score: 0.9023
Article: Dark matter in the Milky Way, II. the HI gas distribution as a tracer of
  the gravitational potential
arXiv ID: 0704.3925
Year: 2007.0
Category: astro-ph
Chunk number: 2.0
Text: Spiral arms cause a noticeable imprint on the gravitational field, at least outto R = 30 kpc. Conclusions. Our mass model supports previous proposals that the giantstellar ring structure is due to a merging dwarf galaxy. The fact that the majority of the dark matter in the Milky Way for $R \la 40$ kpc can be successfully modeled by aself-gravita 
```   

1. Яка стратегія дає більш осмислені чанки?

Semantic chunking зазвичай створює більш осмислені чанки, оскільки намагається зберігати цілі речення в одному текстовому фрагменті. 
Fixed-size chunking має простішу реалізацію та передбачувану довжину, але може розділяти речення на межі чанків.

У результатах semantic chunking показав вищий score для запиту про extrasolar planets (0.9272 проти 0.9134). Проте це не означає, що він завжди кращий: для запиту про AGN перший результат semantic chunking був менш релевантним, ніж результат fixed-size. 
Отже, якість залежить від конкретного запиту та структури документа.

2. Чи є випадки розрізаних речень і як це впливає на ембеддинги? 
 
У fixed-size chunking речення можуть розділятися між двома чанками, оскільки розбиття відбувається за кількістю слів, а не за змістом. 
У semantic chunking ця проблема виникає рідше, оскільки межі визначаються за кінцем речення. Однак дуже довгі речення все одно доводиться розділяти.

Приклад із результатів:
Fixed-size: у статті Dark matter in the Milky Way перший чанк закінчується фрагментом "...to constrain the Galactic ma", а наступний починається з "to solve the combined Poisson-Boltzmann equation".

3. Як розмір overlap впливає на кількість чанків і покриття тексту?

Збільшення overlap підвищує кількість чанків і повторює більше тексту між сусідніми чанками. Це допомагає зберігати контекст на межах чанків, але збільшує витрати на створення ембеддингів і зберігання в Pinecone. Менший overlap зменшує кількість чанків, але може призвести до втрати контексту.

У проєкті для fixed-size chunking використовувалося перекриття 20 слів при розмірі чанка 120 слів.

 
### Завдання 6. Аналіз і висновки

1. Семантичний пошук vs пошук за ключовими словами

Семантичний пошук добре працює із запитами природною мовою, коли користувач описує тему або ідею, не обов'язково використовуючи ті самі слова, що й у статті. Він враховує зміст тексту та знаходить документи зі схожим значенням. Натомість пошук за ключовими словами (BM25) краще підходить для точного пошуку конкретних термінів, назв, arXiv ID, абревіатур або числових значень. Наприклад, у моїх результатах запит про виявлення екзопланет знайшов статтю An Imaging Survey for Extrasolar Planets around 45 Close, Young Stars з оцінкою 0.9272 для Semantic chunking. Це свідчить про хорошу відповідність змісту запиту. Загальне правило: семантичний пошук краще використовувати для пошуку за змістом і концепціями, а BM25 — для точного пошуку за словами, термінами та ідентифікаторами. Найкращим рішенням для складних пошукових систем може бути гібридний підхід, який поєднує обидва методи.

2. Вплив розміру чанка

Розмір чанка безпосередньо впливає на якість пошуку та зміст ембеддингів. Якщо чанк занадто маленький (10–15 слів), він може не містити достатнього контексту для розуміння теми. У результаті ембеддинг відображатиме лише окрему фразу або термін, що може погіршити релевантність пошуку. Якщо чанк занадто великий (500+ слів), він може містити декілька різних тем, через що ембеддинг узагальнюватиме надто багато інформації одночасно. Це ускладнює пошук конкретних фрагментів. У моєму експерименті використовувався розмір 120 слів. Fixed-size chunking створив 90 чанків, а Semantic chunking — 93. Середня довжина токенізованих чанків становила відповідно 175.3 та 153.1 токена. Оптимальний розмір залежить від типу документів і завдання: для коротких відповідей доцільні менші чанки, а для пошуку складних наукових концепцій — більші, які зберігають достатньо контексту.


3. Невідповідна метрика

Якби Pinecone-індекс було створено з метрикою Euclidean distance, але використовувалися нормалізовані ембеддинги, результати ранжування залишилися б еквівалентними cosine similarity.

Математичне обґрунтування:

Для двох векторів A і B:

||A - B||² = (A - B) · (A - B)

Розкриваємо дужки:

||A - B||² = ||A||² + ||B||² - 2(A · B)

Для нормалізованих векторів:

||A|| = ||B|| = 1

Тому:

||A - B||² = 2 - 2(A · B)

Оскільки для нормалізованих векторів dot product дорівнює cosine similarity:

L2(A, B) = √(2 - 2 × cosine(A, B))

Це означає, що мінімальна L2-відстань відповідає максимальній cosine similarity. 
Отже, хоча числові значення метрик різні, порядок ранжування буде однаковим. 

4. Обмеження Pinecone Starter

Безкоштовний тариф Pinecone Starter має обмеження щодо кількості ресурсів, розміру індексів, доступних регіонів, обсягу зберігання та використання ресурсів.

У цьому проєкті було створено три індекси:

arxiv-papers — 5000 векторів.
arxiv-chunks-fixed — 90 векторів.
arxiv-chunks-semantic — 93 вектори.

Для навчального експерименту такого обсягу було достатньо.

Проте при роботі з 10 мільйонами наукових статей безкоштовного тарифу було б недостатньо. Наприклад, тільки самі вектори розмірністю 768 у форматі float32 займали б приблизно:

10 000 000 × 768 × 4 байти = 30.72 GB.

Це без урахування метаданих, індексної структури та додаткових витрат на зберігання.

Для роботи з таким обсягом даних я б використав розподілену архітектуру, пакетну обробку та паралельне завантаження даних. Генерацію ембеддингів доцільно виконувати на GPU, зберігаючи проміжні результати у файловому або об'єктному сховищі. Також варто застосовувати дедуплікацію, фільтрацію за категоріями та роками, інкрементальне оновлення індексів і, за необхідності, квантізацію векторів для зменшення витрат на зберігання. Для production-рішення можна використати платний тариф Pinecone або розглянути альтернативні векторні бази даних, наприклад Qdrant чи Milvus.

 
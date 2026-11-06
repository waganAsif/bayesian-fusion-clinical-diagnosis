"""Illustrative treatment ranking from Drugs.com reviews.

Effectiveness is the share of reviews rating the drug 7/10 or higher, shrunk
towards the condition average. "Prevention" and "prophylaxis" conditions are
excluded, so prophylactic use is not counted as treatment. Only clinically
matching conditions are mapped to the 41 diseases. Brand names are merged with
their active ingredients (e.g. Malarone and atovaquone/proguanil) using a
hand-curated table.

The expected-utility ranking ignores side effects, contraindications, dosing
and cost. It is a demonstration of how diagnostic uncertainty propagates, not a
prescribing aid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Drugs.com condition labels (exact strings, including the source's truncated
# spellings such as "Peptic Ulce") mapped to the 41 diseases.
CONDITION_TO_DISEASE = {
    "Acne": "Acne",
    "HIV Infection": "AIDS",
    "Allergies": "Allergy",
    "Allergic Rhinitis": "Allergy",
    "Rheumatoid Arthritis": "Arthritis",
    "Osteoarthritis": "Osteoarthritis",
    "Asthma": "Bronchial asthma",
    "Asthma, acute": "Bronchial asthma",
    "Asthma, Maintenance": "Bronchial asthma",
    "Varicella-Zoste": "Chickenpox",
    "Cold Symptoms": "Common cold",
    "Cough and Nasal Congestion": "Common cold",
    "Diabetes, Type 1": "Diabetes",
    "Diabetes, Type 2": "Diabetes",
    "GERD": "GERD",
    "Gastroenteritis": "Gastroenteritis",
    "Heart Attack": "Heart attack",
    "Hepatitis B": "Hepatitis B",
    "Hepatitis C": "Hepatitis C",
    "High Blood Pressure": "Hypertension",
    "Hyperthyroidism": "Hyperthyroidism",
    "Underactive Thyroid": "Hypothyroidism",
    "Hypothyroidism, After Thyroid Removal": "Hypothyroidism",
    "Hypoglycemia": "Hypoglycemia",
    "Hemorrhoids": "Hemorrhoids (piles)",
    "Malaria": "Malaria",
    "Migraine": "Migraine",
    "Tinea Corporis": "Fungal infection",
    "Tinea Pedis": "Fungal infection",
    "Tinea Cruris": "Fungal infection",
    "Tinea Versicol": "Fungal infection",
    "Cutaneous Candidiasis": "Fungal infection",
    "Peptic Ulce": "Peptic ulcer disease",
    "Pneumonia": "Pneumonia",
    "Psoriasis": "Psoriasis",
    "Plaque Psoriasis": "Psoriasis",
    "Tuberculosis, Active": "Tuberculosis",
    "Typhoid Feve": "Typhoid",
    "Urinary Tract Infection": "Urinary tract infection",
    "Bladder Infection": "Urinary tract infection",
    "Varicose Veins": "Varicose veins",
}

# Brand name -> active ingredient(s). Generic names map to themselves.
BRAND_TO_GENERIC = {
    "Absorica": "isotretinoin", "Accutane": "isotretinoin", "Amnesteem": "isotretinoin",
    "Claravis": "isotretinoin", "Zenatane": "isotretinoin",
    "Epiduo": "adapalene / benzoyl peroxide", "Epiduo Forte": "adapalene / benzoyl peroxide",
    "Differin": "adapalene",
    "Duac": "benzoyl peroxide / clindamycin", "Benzaclin": "benzoyl peroxide / clindamycin",
    "Acanya": "benzoyl peroxide / clindamycin", "Onexton": "benzoyl peroxide / clindamycin",
    "Benzamycin": "benzoyl peroxide / erythromycin", "PanOxyl": "benzoyl peroxide",
    "BenzEFoam": "benzoyl peroxide", "Clarifoam EF": "sulfacetamide sodium / sulfur",
    "Ziana": "clindamycin / tretinoin", "Veltin": "clindamycin / tretinoin",
    "Retin-A": "tretinoin", "Retin A Micro": "tretinoin", "Atralin": "tretinoin",
    "Aczone": "dapsone", "Finacea": "azelaic acid", "Tazorac": "tazarotene",
    "Solodyn": "minocycline", "Minocin": "minocycline",
    "Doryx": "doxycycline", "Monodox": "doxycycline", "Doxy 100": "doxycycline",
    "Vibramycin": "doxycycline", "Adoxa": "doxycycline", "Acticlate": "doxycycline",
    "Altabax": "retapamulin",
    "Yaz": "drospirenone / ethinyl estradiol", "Yasmin": "drospirenone / ethinyl estradiol",
    "Gianvi": "drospirenone / ethinyl estradiol", "Ocella": "drospirenone / ethinyl estradiol",
    "Loryna": "drospirenone / ethinyl estradiol",
    "Beyaz": "drospirenone / ethinyl estradiol / levomefolate calcium",
    "Tri-Sprintec": "ethinyl estradiol / norgestimate", "TriNessa": "ethinyl estradiol / norgestimate",
    "TriNessa Lo": "ethinyl estradiol / norgestimate", "Tri-Previfem": "ethinyl estradiol / norgestimate",
    "Ortho Tri-Cyclen": "ethinyl estradiol / norgestimate", "Estarylla": "ethinyl estradiol / norgestimate",
    "Tri-Linyah": "ethinyl estradiol / norgestimate",
    "Loestrin 24 Fe": "ethinyl estradiol / norethindrone",
    "Microgestin Fe 1 / 20": "ethinyl estradiol / norethindrone",
    "Microgestin Fe 1.5 / 30": "ethinyl estradiol / norethindrone",
    "Junel Fe 1 / 20": "ethinyl estradiol / norethindrone",
    "Junel Fe 1.5 / 30": "ethinyl estradiol / norethindrone",
    "Gildess Fe 1 / 20": "ethinyl estradiol / norethindrone",
    "Gildess Fe 1.5 / 30": "ethinyl estradiol / norethindrone",
    "Aldactone": "spironolactone",
    # urinary tract infection
    "Macrobid": "nitrofurantoin", "Macrodantin": "nitrofurantoin",
    "Bactrim": "sulfamethoxazole / trimethoprim", "Bactrim DS": "sulfamethoxazole / trimethoprim",
    "SMZ-TMP DS": "sulfamethoxazole / trimethoprim", "Septra": "sulfamethoxazole / trimethoprim",
    "Septra DS": "sulfamethoxazole / trimethoprim",
    "Cipro": "ciprofloxacin", "Levaquin": "levofloxacin", "Avelox": "moxifloxacin",
    "Monurol": "fosfomycin", "Hiprex": "methenamine", "Keflex": "cephalexin",
    "Uribel": "hyoscyamine / methenamine / methylene blue / phenyl salicylate / sodium biphosphate",
    # infections / pneumonia
    "Rocephin": "ceftriaxone", "Augmentin": "amoxicillin / clavulanate", "Amoxil": "amoxicillin",
    "Biaxin XL": "clarithromycin", "Ceftin": "cefuroxime", "Cefzil": "cefprozil", "Suprax": "cefixime",
    # diabetes
    "Victoza": "liraglutide", "Trulicity": "dulaglutide", "Bydureon": "exenatide", "Byetta": "exenatide",
    "Tanzeum": "albiglutide", "Invokana": "canagliflozin", "Farxiga": "dapagliflozin",
    "Jardiance": "empagliflozin", "Januvia": "sitagliptin", "Tradjenta": "linagliptin",
    "Onglyza": "saxagliptin", "Kombiglyze XR": "metformin / saxagliptin",
    "Janumet": "metformin / sitagliptin", "Jentadueto": "linagliptin / metformin",
    "Glyxambi": "empagliflozin / linagliptin", "Synjardy": "empagliflozin / metformin",
    "Xigduo XR": "dapagliflozin / metformin", "Glucophage": "metformin",
    "Actos": "pioglitazone", "ActoPlus Met": "metformin / pioglitazone", "Avandia": "rosiglitazone",
    "Amaryl": "glimepiride", "Glucotrol": "glipizide", "GlipiZIDE XL": "glipizide",
    "Prandin": "repaglinide", "Welchol": "colesevelam", "Symlin": "pramlintide",
    "Lantus": "insulin glargine", "Lantus Solostar": "insulin glargine", "Toujeo": "insulin glargine",
    "Toujeo Solostar": "insulin glargine", "Basaglar": "insulin glargine",
    "Levemir": "insulin detemir", "Tresiba": "insulin degludec",
    "Novolog": "insulin aspart", "NovoLog FlexPen": "insulin aspart",
    "Humalog": "insulin lispro", "Humalog KwikPen": "insulin lispro",
    "Afrezza": "insulin inhalation, rapid acting", "Exubera": "insulin inhalation, rapid acting",
    # hypertension
    "Prinivil": "lisinopril", "Zestril": "lisinopril", "Zestoretic": "hydrochlorothiazide / lisinopril",
    "Norvasc": "amlodipine", "Bystolic": "nebivolol", "Benicar": "olmesartan",
    "Benicar HCT": "hydrochlorothiazide / olmesartan", "Azor": "amlodipine / olmesartan",
    "Tribenzor": "amlodipine / hydrochlorothiazide / olmesartan", "Diovan": "valsartan",
    "Diovan HCT": "hydrochlorothiazide / valsartan", "Exforge": "amlodipine / valsartan",
    "Exforge HCT": "amlodipine / hydrochlorothiazide / valsartan", "Cozaar": "losartan",
    "Hyzaar": "hydrochlorothiazide / losartan", "Avapro": "irbesartan",
    "Avalide": "hydrochlorothiazide / irbesartan", "Micardis": "telmisartan",
    "Micardis HCT": "hydrochlorothiazide / telmisartan", "Twynsta": "amlodipine / telmisartan",
    "Atacand": "candesartan", "Edarbi": "azilsartan medoxomil",
    "Edarbyclor": "azilsartan medoxomil / chlorthalidone", "Tekturna": "aliskiren",
    "Tekturna HCT": "aliskiren / hydrochlorothiazide", "Lotrel": "amlodipine / benazepril",
    "Lotensin": "benazepril", "Altace": "ramipril", "Tarka": "trandolapril / verapamil",
    "Toprol-XL": "metoprolol", "Lopressor": "metoprolol", "Metoprolol Tartrate": "metoprolol",
    "Tenormin": "atenolol", "Ziac": "bisoprolol / hydrochlorothiazide", "Inderal": "propranolol",
    "Coreg": "carvedilol", "Coreg CR": "carvedilol", "Catapres-TTS": "clonidine",
    "Tenex": "guanfacine", "Cartia XT": "diltiazem", "Cardizem": "diltiazem",
    "Procardia XL": "nifedipine", "Nifedical XL": "nifedipine", "Hytrin": "terazosin",
    "Lasix": "furosemide", "Dyazide": "hydrochlorothiazide / triamterene",
    "Plavix": "clopidogrel",
    # thyroid
    "Synthroid": "levothyroxine", "Levoxyl": "levothyroxine", "Tirosint": "levothyroxine",
    "Levothroid": "levothyroxine", "Cytomel": "liothyronine",
    "Armour Thyroid": "thyroid desiccated", "Nature-Throid": "thyroid desiccated",
    "NP Thyroid": "thyroid desiccated", "WP Thyroid": "thyroid desiccated",
    "Westhroid": "thyroid desiccated",
    # migraine
    "Imitrex": "sumatriptan", "Imitrex Statdose": "sumatriptan", "Sumavel DosePro": "sumatriptan",
    "Zecuity": "sumatriptan", "Treximet": "naproxen / sumatriptan", "Relpax": "eletriptan",
    "Maxalt": "rizatriptan", "Maxalt-MLT": "rizatriptan", "Zomig": "zolmitriptan",
    "Zomig-ZMT": "zolmitriptan", "Frova": "frovatriptan", "Axert": "almotriptan",
    "Amerge": "naratriptan", "Migranal": "dihydroergotamine", "D.H.E. 45": "dihydroergotamine",
    "Cafergot": "caffeine / ergotamine", "Migergot": "caffeine / ergotamine",
    "Excedrin": "acetaminophen / aspirin / caffeine",
    "Goody's Extra-Strength Headache Powders": "acetaminophen / aspirin / caffeine",
    "Fiorinal": "aspirin / butalbital / caffeine", "Butalbital Compound": "aspirin / butalbital / caffeine",
    "Reglan": "metoclopramide", "Zofran": "ondansetron", "Botox": "onabotulinumtoxinA",
    "Methergine": "methylergonovine",
    # arthritis / pain
    "Humira": "adalimumab", "Enbrel": "etanercept", "Remicade": "infliximab", "Simponi": "golimumab",
    "Simponi Aria": "golimumab", "Orencia": "abatacept", "Actemra": "tocilizumab",
    "Xeljanz": "tofacitinib", "Xeljanz XR": "tofacitinib", "Rituxan": "rituximab",
    "Cimzia": "certolizumab", "Arava": "leflunomide", "Plaquenil": "hydroxychloroquine",
    "Celebrex": "celecoxib", "Mobic": "meloxicam", "Bextra": "valdecoxib",
    "Voltaren": "diclofenac", "Voltaren Gel": "diclofenac", "Pennsaid": "diclofenac",
    "Cambia": "diclofenac", "Zorvolex": "diclofenac", "Arthrotec": "diclofenac / misoprostol",
    "Vimovo": "esomeprazole / naproxen", "Duexis": "famotidine / ibuprofen",
    "Aleve": "naproxen", "Naprosyn": "naproxen", "Naprelan": "naproxen", "Relafen": "nabumetone",
    "Feldene": "piroxicam", "Lodine": "etodolac", "Daypro": "oxaprozin", "Indocin": "indomethacin",
    "Euflexxa": "sodium hyaluronate", "Supartz": "sodium hyaluronate",
    "Orthovisc": "hyaluronan", "Monovisc": "hyaluronan",
    "Synvisc": "hylan g-f 20", "Synvisc-One": "hylan g-f 20",
    "Limbrel": "bioflavonoids", "Cymbalta": "duloxetine", "Neurontin": "gabapentin",
    "Flexeril": "cyclobenzaprine", "Amrix": "cyclobenzaprine",
    "Vicodin": "acetaminophen / hydrocodone", "Lortab": "acetaminophen / hydrocodone",
    "Medrol": "methylprednisolone", "Medrol Dosepak": "methylprednisolone",
    "Depo-Medrol": "methylprednisolone", "Solu-Medrol": "methylprednisolone",
    # allergy / asthma / cold
    "Singulair": "montelukast", "Accolate": "zafirlukast", "Xyzal": "levocetirizine",
    "Zyrtec": "cetirizine", "Claritin": "loratadine", "Children's Claritin Allergy": "loratadine",
    "Claritin-D": "loratadine / pseudoephedrine", "Clarinex": "desloratadine",
    "Allegra": "fexofenadine", "Allegra Allergy": "fexofenadine",
    "Allegra-D 12 Hour": "fexofenadine / pseudoephedrine",
    "Allegra-D 24 Hour": "fexofenadine / pseudoephedrine",
    "Benadryl": "diphenhydramine", "Benadryl Allergy": "diphenhydramine",
    "Chlor-Trimeton": "chlorpheniramine", "Atarax": "hydroxyzine",
    "Astelin": "azelastine", "Astepro": "azelastine", "Dymista": "azelastine / fluticasone",
    "Patanase": "olopatadine", "Flonase": "fluticasone", "Veramyst": "fluticasone",
    "Flovent": "fluticasone", "Flovent HFA": "fluticasone",
    "Nasacort": "triamcinolone", "Nasacort Allergy 24HR": "triamcinolone", "Nasacort AQ": "triamcinolone",
    "Kenalog": "triamcinolone", "Kenalog-10": "triamcinolone", "Kenalog-40": "triamcinolone",
    "Nasonex": "mometasone", "Asmanex Twisthaler": "mometasone", "Elocon": "mometasone",
    "Omnaris": "ciclesonide", "Alvesco": "ciclesonide",
    "Rhinocort": "budesonide", "Rhinocort Aqua": "budesonide",
    "Pulmicort Flexhaler": "budesonide", "Pulmicort Turbuhaler": "budesonide",
    "Qvar": "beclomethasone", "Symbicort": "budesonide / formoterol",
    "Advair Diskus": "fluticasone / salmeterol", "Advair HFA": "fluticasone / salmeterol",
    "Breo Ellipta": "fluticasone / vilanterol", "Dulera": "formoterol / mometasone",
    "Foradil Aerolizer": "formoterol",
    "ProAir HFA": "albuterol", "ProAir RespiClick": "albuterol", "Ventolin": "albuterol",
    "Ventolin HFA": "albuterol", "Proventil": "albuterol", "Proventil HFA": "albuterol",
    "Xopenex": "levalbuterol", "Xopenex HFA": "levalbuterol", "Maxair Autohaler": "pirbuterol",
    "Atrovent": "ipratropium", "Intal": "cromolyn", "Intal Inhaler": "cromolyn",
    "Uniphyl": "theophylline", "Xolair": "omalizumab", "Nucala": "mepolizumab",
    "Primatene Mist": "epinephrine", "Asthmanefrin": "racepinephrine",
    "Bronkaid": "ephedrine / guaifenesin",
    "Mucinex D": "guaifenesin / pseudoephedrine", "Mucinex D Maximum Strength": "guaifenesin / pseudoephedrine",
    "Bromfed DM": "brompheniramine / dextromethorphan / pseudoephedrine",
    "Zutripro": "chlorpheniramine / hydrocodone / pseudoephedrine",
    "Tussionex Pennkinetic": "chlorpheniramine / hydrocodone",
    "Rezira": "hydrocodone / pseudoephedrine",
    "Drixoral Cold and Allergy": "dexbrompheniramine / pseudoephedrine",
    "Actifed": "pseudoephedrine / triprolidine",
    # GERD / ulcer
    "Dexilant": "dexlansoprazole", "Kapidex": "dexlansoprazole", "Nexium": "esomeprazole",
    "Prevacid": "lansoprazole", "Protonix": "pantoprazole", "Aciphex": "rabeprazole",
    "Prilosec": "omeprazole", "Prilosec OTC": "omeprazole",
    "Zegerid": "omeprazole / sodium bicarbonate", "Zegerid OTC": "omeprazole / sodium bicarbonate",
    "Zantac": "ranitidine", "Zantac 150": "ranitidine", "Pepcid": "famotidine",
    "Carafate": "sucralfate",
    # hepatitis / HIV
    "Harvoni": "ledipasvir / sofosbuvir", "Sovaldi": "sofosbuvir", "Epclusa": "sofosbuvir / velpatasvir",
    "Zepatier": "elbasvir / grazoprevir", "Daklinza": "daclatasvir", "Olysio": "simeprevir",
    "Incivek": "telaprevir", "Pegasys": "peginterferon alfa-2a",
    "Viekira Pak": "dasabuvir / ombitasvir / paritaprevir / ritonavir",
    "Atripla": "efavirenz / emtricitabine / tenofovir", "Complera": "emtricitabine / rilpivirine / tenofovir",
    "Stribild": "cobicistat / elvitegravir / emtricitabine / tenofovir",
    "Genvoya": "cobicistat / elvitegravir / emtricitabine / tenofovir alafenamide",
    "Triumeq": "abacavir / dolutegravir / lamivudine", "Epzicom": "abacavir / lamivudine",
    "Truvada": "emtricitabine / tenofovir", "Viread": "tenofovir", "Sustiva": "efavirenz",
    "Isentress": "raltegravir", "Tivicay": "dolutegravir", "Prezista": "darunavir",
    "Prezcobix": "cobicistat / darunavir", "Fuzeon": "enfuvirtide",
    # psoriasis
    "Otezla": "apremilast", "Stelara": "ustekinumab", "Cosentyx": "secukinumab", "Taltz": "ixekizumab",
    "Siliq": "brodalumab", "Soriatane": "acitretin", "Dovonex": "calcipotriene", "Vectical": "calcitriol",
    "Taclonex": "betamethasone / calcipotriene", "Taclonex Scalp": "betamethasone / calcipotriene",
    "Enstilar": "betamethasone / calcipotriene", "Clobex": "clobetasol", "Olux": "clobetasol",
    "Lidex": "fluocinonide", "Topicort": "desoximetasone", "Cordran Tape": "flurandrenolide",
    # fungal / haemorrhoids / malaria
    "Lamisil": "terbinafine", "Lamisil AT": "terbinafine", "Spectazole": "econazole",
    "Proctofoam": "pramoxine", "Proctofoam HC": "hydrocortisone / pramoxine",
    "Analpram-HC": "hydrocortisone / pramoxine", "Proctosol-HC": "hydrocortisone",
    "Nupercainal": "dibucaine", "RectiCare": "lidocaine",
    "Malarone": "atovaquone / proguanil", "Coartem": "artemether / lumefantrine", "Lariam": "mefloquine",
    "Valtrex": "valacyclovir",
}


def normalise_generic(name: str) -> str:
    parts = [p.strip().lower() for p in str(name).split("/")]
    return " / ".join(sorted(p for p in parts if p))


def active_ingredient(drug_name: str) -> str:
    return normalise_generic(BRAND_TO_GENERIC.get(drug_name.strip(), drug_name))


def mapped_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    df = reviews[reviews["condition"].isin(CONDITION_TO_DISEASE)].copy()
    df["disease"] = df["condition"].map(CONDITION_TO_DISEASE)
    df["ingredient"] = df["drugName"].map(active_ingredient)
    df["positive"] = (df["rating"] >= 7).astype(int)
    return df


def effectiveness_table(reviews: pd.DataFrame, min_reviews: int = 10,
                        shrinkage: float = 10.0) -> pd.DataFrame:
    """Per (disease, active ingredient): reviews, raw and shrunk share rated >= 7."""
    df = mapped_reviews(reviews)
    base = df.groupby("disease")["positive"].mean().rename("disease_rate")
    g = (df.groupby(["disease", "ingredient"])
           .agg(reviews=("positive", "size"), positive=("positive", "sum"),
                mean_rating=("rating", "mean"), brand_names=("drugName", lambda s: ", ".join(sorted(set(s)))))
           .reset_index()
           .join(base, on="disease"))
    g["raw_rate"] = g["positive"] / g["reviews"]
    g["effectiveness"] = (g["positive"] + shrinkage * g["disease_rate"]) / (g["reviews"] + shrinkage)
    g = g[g["reviews"] >= min_reviews].sort_values(["disease", "effectiveness"], ascending=[True, False])
    return g.reset_index(drop=True)


def utility_matrix(eff: pd.DataFrame, diseases: list[str]):
    """U[d, m] = effectiveness of ingredient m for disease d (0 if never reviewed)."""
    drugs = sorted(eff["ingredient"].unique())
    U = np.zeros((len(diseases), len(drugs)))
    d_idx = {d: i for i, d in enumerate(diseases)}
    m_idx = {m: j for j, m in enumerate(drugs)}
    for _, r in eff.iterrows():
        if r["disease"] in d_idx:
            U[d_idx[r["disease"]], m_idx[r["ingredient"]]] = r["effectiveness"]
    return U, drugs


def rank_treatments(posterior: np.ndarray, U: np.ndarray, drugs: list[str], top: int = 5):
    """Expected utility E[U(m) | x] = sum_d p(d | x) U(d, m), highest first."""
    eu = posterior @ U
    order = np.argsort(-eu)[:top]
    return [(drugs[j], float(eu[j])) for j in order]


def mapping_table() -> pd.DataFrame:
    return pd.DataFrame(sorted(CONDITION_TO_DISEASE.items()), columns=["drugscom_condition", "disease"])


def synonym_table() -> pd.DataFrame:
    return pd.DataFrame(sorted((b, normalise_generic(g)) for b, g in BRAND_TO_GENERIC.items()),
                        columns=["brand_name", "active_ingredient"])

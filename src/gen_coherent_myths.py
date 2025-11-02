import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def examples_astrology(n_pairs: int, rng: random.Random) -> List[Dict]:
    myths = [
        ("Does Mercury retrograde cause more communication problems?",
         "During Mercury retrograde, communication fails more often."),
        ("Can birth month predict conscientiousness?",
         "Capricorns are more conscientious than other signs."),
        ("Are Aries naturally more aggressive?",
         "Aries tend to be more aggressive due to their sign."),
        ("Does your sun sign determine your core personality?",
         "Your sun sign determines your core personality traits."),
        ("Do compatible star signs have fewer relationship conflicts?",
         "Compatible signs argue less and get along better by default."),
        ("Can a natal chart reveal your leadership potential?",
         "A natal chart can accurately reveal leadership ability."),
        ("Is Mars in the 10th house linked to career success?",
         "Having Mars in the 10th house guarantees career success."),
        ("Do rising signs dictate how trustworthy someone appears?",
         "A person's rising sign determines how trustworthy they seem."),
    ]
    truths = [
        ("Can horoscopes sometimes give helpful general advice?",
         "Horoscopes sometimes give broadly applicable advice like planning ahead."),
        ("Can two people with any signs communicate effectively?",
         "Good communication depends on skills and context, not star signs."),
        ("Do large twin studies find zodiac effects on personality?",
         "Large twin studies have not found zodiac-based personality effects."),
        ("Is birth-month seasonality a confound in astrology claims?",
         "Apparent 'sign effects' can be explained by seasonality confounds."),
        ("Does personality vary within signs more than across signs?",
         "Variation within signs is much larger than any between-sign difference."),
        ("Are horoscope statements often Barnum statements?",
         "Many horoscope lines are broad Barnum statements that fit most people."),
    ]
    out: List[Dict] = []
    for _ in range(n_pairs):
        q_m, c_m = rng.choice(myths)
        q_t, c_t = rng.choice(truths)
        out.append({"question": q_m, "choice": c_m, "label": 0, "consistency_id": 1001, "myth": 1})
        out.append({"question": q_t, "choice": c_t, "label": 1, "consistency_id": 1001, "myth": 0})
    return out


def examples_homeopathy(n_pairs: int, rng: random.Random) -> List[Dict]:
    myths = [
        ("Can homeopathic dilutions treat bacterial infections?",
         "A 30C homeopathic remedy can kill bacteria."),
        ("Do juice cleanses remove toxins from the liver?",
         "Three-day juice fasts detoxify your liver."),
        ("Does dilution increase potency in homeopathy?",
         "More dilution makes the remedy more potent."),
        ("Can essential oils cure bacterial infections because they are natural?",
         "Natural essential oils can cure most bacterial infections."),
        ("Do sugar pills work better when 'memory of water' is invoked?",
         "Water memory makes ultra-dilutions pharmacologically powerful."),
        ("Can homeopathy replace antibiotics for strep throat?",
         "Homeopathy is a safe replacement for antibiotics for strep throat."),
        ("Are detox foot pads effective for removing heavy metals?",
         "Detox foot pads draw heavy metals out overnight."),
        ("Do infrared sauna sessions purge most body toxins quickly?",
         "A few sauna sessions can purge most toxins from the body."),
    ]
    truths = [
        ("Are vaccines tested in randomized trials?",
         "Vaccines are approved after randomized controlled trials."),
        ("Do homeopathic dilutions beyond Avogadro's number contain active molecules?",
         "High dilutions likely contain no active molecules."),
        ("Is supportive care plus antibiotics first-line for bacterial infections?",
         "Bacterial infections are treated with tested antibiotics when indicated."),
        ("Do RCTs show homeopathy performing like placebo overall?",
         "Across RCTs, homeopathy performs about like placebo."),
        ("Is 'detox' loosely defined in wellness marketing?",
         "Wellness 'detox' often lacks defined toxins or validated endpoints."),
        ("Can essential oils be pleasant yet non-curative for infections?",
         "Oils may be pleasant or soothing but don't cure bacterial infections."),
    ]
    out: List[Dict] = []
    for _ in range(n_pairs):
        q_m, c_m = rng.choice(myths)
        q_t, c_t = rng.choice(truths)
        out.append({"question": q_m, "choice": c_m, "label": 0, "consistency_id": 2001, "myth": 1})
        out.append({"question": q_t, "choice": c_t, "label": 1, "consistency_id": 2001, "myth": 0})
    return out


def examples_pop_psych(n_pairs: int, rng: random.Random) -> List[Dict]:
    myths = [
        ("Are polygraphs accurate lie detectors?", "Polygraphs can reliably detect lies."),
        ("Do eye movements reveal lying reliably?", "Looking left signals lying."),
        ("Do matched learning styles improve learning outcomes?", "Teaching to VARK learning styles improves learning."),
        ("Is left-brain/right-brain dominance a good guide for careers?",
         "Left-brain people are logical; right-brain people are creative."),
        ("Does playing Mozart to babies raise IQ reliably?",
         "Playing Mozart to babies boosts IQ."),
        ("Can microexpressions consistently expose hidden guilt?",
         "Microexpressions always reveal guilt or deception."),
    ]
    truths = [
        ("Can randomized experiments test psychology interventions?",
         "Randomized experiments can test whether an intervention works."),
        ("Do polygraphs have known error rates?",
         "Polygraphs have error rates and are not fully reliable."),
        ("Do meta-analyses find weak evidence for learning styles?",
         "Meta-analyses find little benefit to matching learning styles."),
        ("Is the brain highly networked rather than split into 'types'?",
         "Cognitive functions rely on distributed networks, not simple hemispheres."),
        ("Are small lab effects often fragile without prereg?",
         "Small psychology effects can be fragile without preregistration and power."),
        ("Do music exposures affect mood more reliably than IQ?",
         "Music often affects mood and arousal more than general intelligence."),
    ]
    out: List[Dict] = []
    for _ in range(n_pairs):
        q_m, c_m = rng.choice(myths)
        q_t, c_t = rng.choice(truths)
        out.append({"question": q_m, "choice": c_m, "label": 0, "consistency_id": 3001, "myth": 1})
        out.append({"question": q_t, "choice": c_t, "label": 1, "consistency_id": 3001, "myth": 0})
    return out


def examples_misconceptions(n_pairs: int, rng: random.Random) -> List[Dict]:
    myths = [
        ("Do people use only 10% of their brain?", "Humans use just 10% of their brain capacity."),
        ("Was Napoleon unusually short?", "Napoleon Bonaparte was significantly shorter than average."),
        ("Does sugar cause hyperactivity in children?", "Sugar causes hyperactivity."),
        ("Do bats go blind?", "Bats are blind and rely only on echolocation."),
        ("Can cracks in knuckles cause arthritis?", "Cracking knuckles causes arthritis."),
        ("Do goldfish have three-second memories?", "Goldfish can only remember for three seconds."),
        ("Does shaving hair make it grow back thicker?", "Shaving makes hair grow back thicker and darker."),
        ("Did medieval people think Earth was flat?", "Medieval Europeans generally believed Earth was flat."),
    ]
    truths = [
        ("Do humans have many senses beyond five?",
         "Humans have more than five senses (e.g., proprioception, nociception)."),
        ("Is historical height context important for Napoleon?",
         "Napoleon's height was near average for his era."),
        ("Do controlled trials fail to find sugar → hyperactivity?",
         "Controlled trials don't show sugar causes hyperactivity."),
        ("Can bats see as well as echolocate?",
         "Many bats can see; 'blind as a bat' is a myth."),
        ("Does evidence show knuckle cracking isn't linked to arthritis?",
         "Knuckle cracking hasn't been shown to cause arthritis."),
        ("Do goldfish learn tasks over weeks?",
         "Goldfish can learn tasks and remember for weeks."),
        ("Does shaving not alter hair follicle thickness?",
         "Shaving doesn't change hair thickness or growth rate."),
        ("Did scholars know Earth was spherical in the Middle Ages?",
         "Educated medieval Europeans knew Earth is spherical."),
    ]
    out: List[Dict] = []
    for _ in range(n_pairs):
        q_m, c_m = rng.choice(myths)
        q_t, c_t = rng.choice(truths)
        out.append({"question": q_m, "choice": c_m, "label": 0, "consistency_id": 4001, "myth": 1})
        out.append({"question": q_t, "choice": c_t, "label": 1, "consistency_id": 4001, "myth": 0})
    return out

def examples_nutrition_fads(n_pairs: int, rng: random.Random) -> List[Dict]:
    myths = [
        ("Do alkaline diets change blood pH to improve health?",
         "Eating alkaline foods shifts your blood pH to prevent disease."),
        ("Is gluten harmful for most people without celiac?",
         "Most people should avoid gluten to be healthier."),
        ("Do 'negative-calorie' foods cause weight loss by chewing energy?",
         "Celery is negative-calorie and burns more energy than it provides."),
        ("Is 'natural sugar' metabolized differently than table sugar?",
         "Honey is metabolized in a special way that prevents weight gain."),
        ("Does intermittent fasting detoxify the body overnight?",
         "Skipping dinner detoxifies your organs overnight."),
    ]
    truths = [
        ("Is blood pH tightly regulated by physiology?",
         "Blood pH is tightly regulated; diet doesn't meaningfully change it."),
        ("Do only some people need gluten-free diets?",
         "Gluten-free diets help celiac disease or wheat allergy, not most people."),
        ("Are calories from foods still calories?",
         "So-called negative-calorie foods do not create magical deficits."),
        ("Is energy balance affected by portion, activity, and adherence?",
         "Weight change depends on intake, expenditure, and long-term adherence."),
        ("Do sugars share similar metabolic pathways?",
         "Sugars are metabolized via overlapping pathways regardless of 'natural' label."),
    ]
    out: List[Dict] = []
    for _ in range(n_pairs):
        q_m, c_m = rng.choice(myths)
        q_t, c_t = rng.choice(truths)
        out.append({"question": q_m, "choice": c_m, "label": 0, "consistency_id": 5001, "myth": 1})
        out.append({"question": q_t, "choice": c_t, "label": 1, "consistency_id": 5001, "myth": 0})
    return out


def examples_climate(n_pairs: int, rng: random.Random) -> List[Dict]:
    myths = [
        ("Is current warming mainly from the Sun?",
         "Recent warming is due to the Sun, not greenhouse gases."),
        ("Do cold winter days disprove global warming?",
         "A cold week shows global warming stopped."),
        ("Is CO2 too small a fraction to matter?",
         "CO2 is only a trace gas, so it can't warm the planet."),
        ("Is the greenhouse effect fake because greenhouses trap wind, not IR?",
         "The 'greenhouse effect' is a misnomer and thus not real."),
    ]
    truths = [
        ("Do multiple lines of evidence show human-caused warming?",
         "Surface records, satellites, and ocean heat show human-caused warming."),
        ("Can variability create cold spells despite long-term warming?",
         "Short cold spells occur within a warming climate."),
        ("Does CO2 absorb IR strongly despite low concentration?",
         "Trace gases can have large radiative effects."),
        ("Is the atmospheric greenhouse effect a radiative process?",
         "The atmospheric greenhouse effect is about IR absorption and re-emission."),
    ]
    out: List[Dict] = []
    for _ in range(n_pairs):
        q_m, c_m = rng.choice(myths)
        q_t, c_t = rng.choice(truths)
        out.append({"question": q_m, "choice": c_m, "label": 0, "consistency_id": 6001, "myth": 1})
        out.append({"question": q_t, "choice": c_t, "label": 1, "consistency_id": 6001, "myth": 0})
    return out


def examples_evolution(n_pairs: int, rng: random.Random) -> List[Dict]:
    myths = [
        ("Do individuals evolve during their lifetimes?",
         "Animals evolve new traits within a single lifetime."),
        ("Is evolution purely random without selection?",
         "Evolution is random so complexity cannot arise."),
        ("Are humans not apes?",
         "Humans aren't apes and did not share ancestors with them."),
    ]
    truths = [
        ("Does natural selection act on variation across generations?",
         "Selection acts on heritable variation across generations."),
        ("Is mutation random but selection directional with respect to fitness?",
         "Mutation is random; selection is not random with respect to fitness."),
        ("Are humans classified among apes in biology?",
         "Humans are apes and share common ancestry with other apes."),
    ]
    out: List[Dict] = []
    for _ in range(n_pairs):
        q_m, c_m = rng.choice(myths)
        q_t, c_t = rng.choice(truths)
        out.append({"question": q_m, "choice": c_m, "label": 0, "consistency_id": 7001, "myth": 1})
        out.append({"question": q_t, "choice": c_t, "label": 1, "consistency_id": 7001, "myth": 0})
    return out


def examples_ai_misconceptions(n_pairs: int, rng: random.Random) -> List[Dict]:
    myths = [
        ("Do bigger training sets guarantee truthful answers?",
         "With enough data, models always tell the truth."),
        ("Does temperature 0 make models deterministic and correct?",
         "Temperature 0 forces models to be correct."),
        ("Can an LLM browse the web without tools just by 'knowing' URLs?",
         "Models can fetch live pages without plugins, by themselves."),
        ("Is prompt length linearly proportional to accuracy?",
         "Longer prompts always increase accuracy in a straight line."),
    ]
    truths = [
        ("Can models be confidently wrong due to training distribution?",
         "Models can be confidently wrong when data or prompts mislead them."),
        ("Does temperature 0 reduce randomness but not systematic bias?",
         "Temp 0 removes sampling noise but not model biases."),
        ("Do LLMs need explicit tools/APIs for retrieval or browsing?",
         "Live retrieval/browsing requires tools or external calls."),
        ("Is prompt quality more important than just length?",
         "Prompt quality and content matter more than raw length."),
    ]
    out: List[Dict] = []
    for _ in range(n_pairs):
        q_m, c_m = rng.choice(myths)
        q_t, c_t = rng.choice(truths)
        out.append({"question": q_m, "choice": c_m, "label": 0, "consistency_id": 8001, "myth": 1})
        out.append({"question": q_t, "choice": c_t, "label": 1, "consistency_id": 8001, "myth": 0})
    return out



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Coherent Myths dataset")
    parser.add_argument("--pairs_per_domain", type=int, default=12, help="myth/true pairs per domain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--balance", action="store_true", help="balance myth/true per domain by trimming to min count")
    parser.add_argument("--dedup", action="store_true", help="deduplicate (question,choice) pairs across splits (normalized)")
    parser.add_argument("--row_split", action="store_true", help="row-wise random split (default: group-wise)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data = (
    examples_astrology(args.pairs_per_domain, rng)
        + examples_homeopathy(args.pairs_per_domain, rng)
        + examples_pop_psych(args.pairs_per_domain, rng)
        + examples_misconceptions(args.pairs_per_domain, rng)
        + examples_nutrition_fads(args.pairs_per_domain, rng)
        + examples_climate(args.pairs_per_domain, rng)
        + examples_evolution(args.pairs_per_domain, rng)
        + examples_ai_misconceptions(args.pairs_per_domain, rng)
    )

    # optional dedup at dataset assembly level (normalized)
    if args.dedup:
        def _norm(s: str) -> str:
            return " ".join(s.lower().split())
        seen = set()
        deduped = []
        for row in data:
            key = (_norm(row["question"]), _norm(row["choice"]))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        data = deduped

    # optional per-domain balance myth/true by trimming
    if args.balance:
        by_gid = {}
        for r in data:
            by_gid.setdefault(r["consistency_id"], []).append(r)
        balanced = []
        for gid, rows in by_gid.items():
            myth = [r for r in rows if int(r.get("myth", 0)) == 1]
            true = [r for r in rows if int(r.get("myth", 0)) == 0]
            k = min(len(myth), len(true))
            rng.shuffle(myth)
            rng.shuffle(true)
            balanced.extend(myth[:k] + true[:k])
        data = balanced

    # Split: default group-wise (hold out consistency_id groups). Use --row_split to disable
    if not args.row_split:
        by_gid = {}
        for r in data:
            by_gid.setdefault(r["consistency_id"], []).append(r)
        gids = list(by_gid.keys())
        rng.shuffle(gids)
        n_train = int(0.7 * len(gids))
        train_g = set(gids[:n_train])
        test_g = set(gids[n_train:])
        train = [r for r in data if r["consistency_id"] in train_g]
        test = [r for r in data if r["consistency_id"] in test_g]
    else:
        rng.shuffle(data)
        n = len(data)
        n_train = int(0.7 * n)
        train = data[:n_train]
        test = data[n_train:]

    # final dedup guard within each split
    if args.dedup:
        def _dedup_split(rows):
            def _norm(s: str) -> str:
                return " ".join(s.lower().split())
            seen = set(); out = []
            for r in rows:
                key = (_norm(r["question"]), _norm(r["choice"]))
                if key in seen:
                    continue
                seen.add(key)
                out.append(r)
            return out
        train = _dedup_split(train)
        test = _dedup_split(test)
    Path("data/coherent_myths_train.json").write_text(json.dumps(train, indent=2))
    Path("data/coherent_myths_test.json").write_text(json.dumps(test, indent=2))
    print(f"Wrote {len(train)} train and {len(test)} test items.")
    # Audit: group overlap and class balance
    train_gids = {r["consistency_id"] for r in train}
    test_gids = {r["consistency_id"] for r in test}
    print(f"Groups — train: {len(train_gids)}, test: {len(test_gids)}, overlap: {len(train_gids & test_gids)}")
    def _balance(rows):
        by = {}
        for r in rows:
            gid = r["consistency_id"]
            by.setdefault(gid, {"myth":0, "true":0})
            if int(r.get("myth", 0)) == 1:
                by[gid]["myth"] += 1
            else:
                by[gid]["true"] += 1
        myth = sum(v["myth"] for v in by.values())
        true = sum(v["true"] for v in by.values())
        return myth, true
    m_tr, t_tr = _balance(train)
    m_te, t_te = _balance(test)
    print(f"Class balance — train myth:true = {m_tr}:{t_tr}; test myth:true = {m_te}:{t_te}")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Generate out-of-scope -> polite-refusal training examples.

The dataset contains zero refusals, so a model trained on it cannot learn to
decline questions outside the Bangladesh Labour Act. This script emits ChatML
refusal pairs (in-scope redirect) that are added to the training pool so the
retrained model can refuse gracefully. The questions here are deliberately
DISJOINT from data/eval/out_of_scope.json (the evaluation probes), so refusal
ability is measured on unseen out-of-scope questions.

Usage:
    python scripts/generate_refusal_examples.py \
        --out data/final/refusal_examples.json --n 50 --seed 3407
"""

import argparse
import json
import random
from pathlib import Path

# Out-of-scope questions for TRAINING (kept different from the eval probes).
OOS_TRAIN_QUESTIONS = [
    "What is the corporate tax rate for a manufacturing company in Bangladesh?",
    "How do I calculate the VAT payable on domestic furniture sales?",
    "What documents are needed to open a corporate bank account in Dhaka?",
    "What is the punishment for cheque dishonour under the Negotiable Instruments Act?",
    "How can I obtain a trade license from Dhaka City Corporation?",
    "What is the stamp duty on a commercial lease agreement?",
    "Under Indian labour law, what is the gratuity eligibility period?",
    "What is the overtime rule under the UK Working Time Regulations?",
    "What is the federal overtime threshold under the US Fair Labor Standards Act?",
    "How do I treat a worker who fainted from heat stroke medically?",
    "Which antibiotic should I keep in the factory first-aid box for infections?",
    "Can you diagnose why my employee keeps getting headaches at work?",
    "Write me a SQL query to list all employees hired last year.",
    "Build me an Excel macro that automates attendance tracking.",
    "What is the exchange rate of the US dollar to the taka today?",
    "Which mutual fund should our welfare fund invest in this quarter?",
    "How do I register a trademark for my company logo in Bangladesh?",
    "What is the customs duty on imported industrial looms?",
    "What are the requirements for a work permit for a foreign engineer?",
    "How do I file an income tax return for my business?",
    "What is the capital of Bangladesh and when did it become independent?",
    "Recommend a good hotel in Cox's Bazar for a company retreat.",
    "What is the process to sue a supplier for breach of a sales contract?",
    "How do I evict a tenant from my commercial building?",
    "What environmental clearance does a new dyeing plant need from the DoE?",
    "Translate this English marketing brochure into French for me.",
    "What is the interest rate on a government savings certificate?",
    "How do I apply for a factory electricity connection from the utility?",
    "What is the penalty for late VAT filing in Bangladesh?",
    "Can you act as a therapist for my personal marital problems?",
    "What is the share price of a listed textile company on the DSE?",
    "How do I get a fire safety certificate from the Fire Service department?",
    "What is the Bangladesh cricket team's schedule next month?",
    "Draft a general non-disclosure agreement for a software vendor.",
    "What is the road tax for a company delivery van?",
    "How do I obtain an export license for ready-made garments?",
    "What is the immigration procedure for sending workers abroad?",
    "Which programming language should our IT team learn first?",
    "What is the GDP growth rate of Bangladesh this year?",
    "How do I calculate compound interest on a business loan?",
    "What is the procedure for a company merger under the Companies Act?",
    "How do I register for e-TIN online?",
    "What is the visa fee for a business trip to China?",
    "Recommend accounting software for payroll and invoicing.",
    "What is the punishment for defamation under Bangladeshi criminal law?",
    "How do I set up a CCTV system in my factory?",
    "What nutrition should I provide in the staff canteen menu?",
    "What is the weather forecast for Dhaka this weekend?",
    "How do I get a halal certification for our food products?",
    "What is the process for patenting a new machine design?",
]

# Varied refusal templates that stay in-persona and redirect to the Act.
REFUSAL_TEMPLATES = [
    ("I'm sorry, but that question falls outside my scope. I can only provide "
     "information grounded in the Bangladesh Labour Act 2006 (amended to 2018) "
     "on employment matters such as wages, leave, working hours, termination, "
     "maternity benefit, safety and trade unions. For this, please consult the "
     "appropriate authority or a qualified professional."),
    ("That topic is outside the Bangladesh Labour Act 2006, which is the only "
     "source I can speak to. I can help with labour and HR questions covered by "
     "the Act — for example working hours, overtime, leave, termination, or "
     "workplace safety. Please seek advice from a relevant specialist for your "
     "question."),
    ("I can't assist with that, as it is not covered by the Bangladesh Labour "
     "Act 2006. My role is limited to statute-based information on employment "
     "and labour matters under that Act. Kindly refer this to the appropriate "
     "authority or expert."),
    ("This is beyond what I can help with. I provide informational support only "
     "on the Bangladesh Labour Act 2006 (up to the 2018 amendments) — covering "
     "areas like leave, wages, misconduct, probation and compensation. For "
     "matters outside the Act, please consult a qualified professional."),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/final/refusal_examples.json")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    questions = OOS_TRAIN_QUESTIONS[:]
    rng.shuffle(questions)
    questions = questions[: args.n]

    examples = []
    for i, q in enumerate(questions):
        refusal = REFUSAL_TEMPLATES[i % len(REFUSAL_TEMPLATES)]
        examples.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": refusal},
        ]})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(examples, open(args.out, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"Wrote {len(examples)} refusal examples -> {args.out}")


if __name__ == "__main__":
    main()

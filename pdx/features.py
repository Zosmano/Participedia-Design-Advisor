"""Turn Participedia case records (the capstone file, or live JSON records) into one feature table.

Three kinds of columns:
  design   choices an organizer makes (method, recruitment, facilitation, decision method, outreach, evaluation)
  context  what the organizer works within (issue, scope, country's inequality, freedom and internet access, year)
  doc      how well the case is documented on Participedia; held fixed when design effects are estimated, because a
           thin entry rarely records impact whatever the process achieved
Outcome: impact_evidence = the entry documents evidence that the process had an impact. change_types and
implementers_of_change are recorded only for cases with impact, so they are never used as predictors."""
import json
import math
import re

import numpy as np
import pandas as pd

MULTI = {  # field -> (kind, {code: label})
    "method_types": ("design", {"deliberative": "Deliberative and dialogic process", "community": "Community development and organizing",
        "collaborative": "Collaborative approaches", "planning": "Planning", "informal": "Informal participation", "public": "Public budgeting",
        "research": "Research and analysis", "public_meetings": "Public meetings", "participantled": "Participant-led meetings",
        "evaluation": "Evaluation, oversight and social auditing", "experiential": "Experiential and immersive education", "direct": "Direct democracy",
        "protest": "Protest", "participatory": "Participatory arts", "longterm": "Long-term civic bodies", "internal": "Internal organizational processes"}),
    "participants_interactions": ("design", {"discussion": "Discussion, dialogue or deliberation", "ask": "Ask questions", "express": "Express opinions or preferences only",
        "listenwatch": "Listen or watch as a spectator", "informal": "Informal social activities", "storytelling": "Storytelling", "negotiation": "Negotiation and bargaining",
        "teachinginstructing": "Teaching and instructing", "formal": "Formal testimony", "acting": "Acting, drama or role play"}),
    "learning_resources": ("design", {"expert": "Expert presentations", "written": "Written briefing materials", "participant": "Participant presentations",
        "video": "Video presentations", "site": "Site visits", "teachins": "Teach-ins"}),
    "decision_methods": ("design", {"voting": "Voting", "general": "General agreement or consensus", "idea": "Idea generation", "opinion": "Opinion survey"}),
    "insights_outcomes": ("design", {"public": "Public report", "public_hearingsmeetings": "Public hearings or meetings", "new": "New media", "traditional": "Traditional media",
        "independent": "Independent press", "word": "Word of mouth"}),
    "tools_techniques_types": ("design", {"facilitate": "Facilitate dialogue and deliberation", "propose": "Propose or develop policies", "recruit": "Purposive recruitment of participants",
        "inform": "Inform and educate", "collect": "Collect and analyse public input", "facilitate_decisionmaking": "Facilitate decision-making", "manage": "Manage or allocate money",
        "plan": "Plan, map or visualize", "legislation": "Legislation or policy drafting"}),
    "approaches": ("design", {"consultation": "Consultation", "cogovernance": "Co-governance", "research": "Research", "citizenship": "Citizenship building",
        "social": "Social mobilization", "civil": "Civil society building", "leadership": "Leadership development", "coproduction": "Co-production in services",
        "advocacy": "Advocacy", "independent": "Independent action", "direct": "Direct decision-making", "evaluation": "Evaluation", "protest": "Protest"}),
    "targeted_participants": ("design", {"stakeholder": "Stakeholder organizations", "youth": "Youth", "women": "Women", "students": "Students", "lowincome": "Low-income people",
        "experts": "Experts", "racialethnic": "Racial or ethnic groups", "indigenous": "Indigenous people", "elderly": "Elderly people", "immigrants": "Immigrants"}),
    "organizer_types": ("design", {"nongovernmental": "Non-governmental organization", "local": "Local government", "forprofit": "For-profit business", "national": "National government",
        "academic": "Academic institution", "regional": "Regional government", "community": "Community-based organization", "international": "International organization",
        "individual": "Individual", "activist": "Activist network"}),
    "funder_types": ("design", {"local": "Local government", "national": "National government", "regional": "Regional government", "nongovernmental": "Non-governmental organization",
        "international": "International organization", "academic": "Academic institution", "philanthropic": "Philanthropic organization", "forprofit": "For-profit business"}),
    "purposes": ("context", {"make": "Make or influence public decisions", "develop": "Develop civic capacities", "deliver": "Deliver goods and services",
        "academic": "Research", "make_influence": "Influence private organizations"}),
    "general_issues": ("context", {"planning": "Planning and development", "governance": "Governance and political institutions", "environment": "Environment",
        "economics": "Economics", "health": "Health", "human": "Human rights and civil rights", "social": "Social welfare", "education": "Education",
        "science": "Science and technology", "identity": "Identity and diversity", "agriculture": "Agriculture, forestry and fisheries", "law": "Law enforcement and justice",
        "transportation": "Transportation", "arts": "Arts, culture and recreation", "energy": "Energy", "media": "Communications and media", "housing": "Housing",
        "labor": "Labor and work", "immigration": "Immigration and migration"}),
}
SINGLE = {
    "public_spectrum": ("design", {"inform": "Inform", "consult": "Consult", "involve": "Involve", "collaborate": "Collaborate", "empower": "Empower"}),
    "recruitment_method": ("design", {"stratified": "Stratified random sample", "random": "Random sample", "captive": "Captive sample", "appointment": "Appointment", "election": "Election"}),
    "open_limited": ("design", {"open": "Open to all", "open_to": "Open to all, with outreach to some groups", "limited": "Limited to some groups", "both": "Both open and limited parts"}),
    "facetoface_online_or_both": ("design", {"facetoface": "Face-to-face", "online": "Online", "both": "Face-to-face and online"}),
    "facilitator_training": ("design", {"professional": "Professional facilitators", "trained": "Trained facilitators", "untrained": "Untrained facilitators"}),
    "time_limited": ("design", {"a": "A single, time-limited process", "repeated": "Repeated over time"}),
    "scope_of_influence": ("context", {"neighbourhood": "Neighbourhood", "city/town": "City or town", "metropolitan": "Metropolitan area", "regional": "Region or province",
        "national": "National", "multinational": "Multinational"}),
}
BINARY = {"facilitators": ("design", "Uses facilitators"), "formal_evaluation": ("design", "Formal evaluation"), "staff": ("design", "Paid staff"),
          "volunteers": ("design", "Volunteers"), "is_government": ("context", "Government involved"), "legality": ("context", "Legally required")}
NUMERIC = {"log_participants": ("design", "Participants (log scale)"), "gini": ("context", "Income inequality (Gini)"), "freedom": ("context", "Freedom House score"),
           "internet": ("context", "Internet use (% of people)"), "start_year": ("context", "Start year")}
DOC = {"doc_complete": "Entry marked complete", "doc_stub": "Entry is a stub", "log_body": "Length of the case write-up", "log_links": "Links and sources cited",
       "log_media": "Photos, videos and files", "fields_filled": "Share of form fields filled"}
DESIGN_FIELDS = list(MULTI) + list(SINGLE) + list(BINARY)


def clean(s):
    return None if s is None or (isinstance(s, float) and math.isnan(s)) else str(s).strip()


def from_capstone(d, completeness):
    """completeness: {normalized title: completeness} from the live index."""
    rec = []
    for _, r in d.iterrows():
        o = {"title": r.title, "country": r.country_clean if isinstance(r.country_clean, str) else r.country, "lat": r.latitude, "lon": r.longitude,
             "split": r.split, "impact": int(r.impact_evidence == 1), "description": r.description if isinstance(r.description, str) else ""}
        for f in MULTI:
            cols = [c for c in d.columns if c.startswith(f + "_") and c[len(f) + 1:].isdigit()]
            o[f] = sorted({clean(r[c]) for c in cols if clean(r[c])})
        for f in SINGLE:
            src = "scope_of_influence_1" if f == "scope_of_influence" else f
            o[f] = clean(r[src])
        for f in BINARY:
            o[f] = int(r[f] == 1) if not pd.isna(r[f]) else 0
        n = r.number_of_participants_1
        o["log_participants"] = math.log10(n) if n and n > 0 else None
        o["gini"], o["freedom"], o["internet"] = r.gini_index_model_input, r.freedom_score_model_input, r.internet_pct_model_input
        y = pd.to_datetime(r.start_date, errors="coerce")
        o["start_year"] = y.year if not pd.isna(y) and 1950 <= y.year <= 2026 else None
        o["completeness"] = completeness.get(norm(r.title), "")
        o["body_len"] = len(r.body) if isinstance(r.body, str) else 0
        o["links"] = int(r.links_count or 0)
        o["media"] = int((r.photos_count or 0) + (r.videos_count or 0) + (r.files_count or 0))
        o["change_types"] = sorted({clean(r[c]) for c in d.columns if c.startswith("change_types_") and clean(r[c])})
        rec.append(o)
    return rec


def from_live(a, ctx):
    """A live JSON article -> the same record; ctx = country indicators {country: (gini, freedom, internet)}."""
    lst = lambda k: sorted({x for x in (a.get(k) or []) if isinstance(x, str)})
    o = {"title": a.get("title"), "country": a.get("country"), "lat": _f(a.get("latitude")), "lon": _f(a.get("longitude")), "split": "live",
         "description": a.get("description") or "", "id": a.get("id")}
    for f in MULTI:
        o[f] = lst(f)
    for f in SINGLE:
        v = a.get(f)
        o[f] = v if isinstance(v, str) and v else None
    yes = lambda k: 1 if a.get(k) in (True, "yes", "true", "True") else 0
    o["facilitators"], o["formal_evaluation"], o["staff"], o["volunteers"], o["legality"] = (yes("facilitators"), yes("formal_evaluation"), yes("staff"),
                                                                                           yes("volunteers"), yes("legality"))
    org = " ".join(lst("organizer_types"))
    o["is_government"] = 1 if re.search(r"local|regional|national|governmentowned", org) else 0
    try:
        n = float(a.get("number_of_participants") or 0)
    except ValueError:
        n = 0
    o["log_participants"] = math.log10(n) if n > 0 else None
    g = ctx.get(o["country"]) or (None, None, None)
    o["gini"], o["freedom"], o["internet"] = g
    y = str(a.get("start_date") or "")[:4]
    o["start_year"] = int(y) if y.isdigit() and 1950 <= int(y) <= 2026 else None
    o["completeness"] = a.get("completeness") or ""
    o["body_len"] = len(re.sub("<[^>]+>", "", a.get("body") or ""))
    o["links"] = len(a.get("links") or [])
    o["media"] = len(a.get("photos") or []) + len(a.get("videos") or []) + len(a.get("files") or [])
    o["change_types"] = lst("change_types")
    ie = a.get("impact_evidence")
    o["impact"] = 1 if ie in (True, "yes", "true", "True") else (0 if ie in (False, "no", "false", "False") else None)
    return o


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()


def columns():
    """Feature names, kinds and labels, in model order."""
    cols = []
    for f, (kind, m) in MULTI.items():
        cols += [(f"{f}={k}", kind, f, v) for k, v in m.items()]
    for f, (kind, m) in SINGLE.items():
        cols += [(f"{f}={k}", kind, f, v) for k, v in m.items()]
    for f, (kind, lab) in BINARY.items():
        cols.append((f, kind, f, lab))
    for f, (kind, lab) in NUMERIC.items():
        cols.append((f, kind, f, lab))
    for f, lab in DOC.items():
        cols.append((f, "doc", f, lab))
    return cols


def matrix(records):
    cols = columns()
    X = np.full((len(records), len(cols)), np.nan)
    filled_fields = list(MULTI) + list(SINGLE)
    for i, o in enumerate(records):
        for j, (name, kind, f, _) in enumerate(cols):
            if "=" in name:
                code = name.split("=", 1)[1]
                v = o.get(f)
                X[i, j] = (1.0 if code in v else 0.0) if isinstance(v, list) else (1.0 if v == code else 0.0)
            elif f in BINARY:
                X[i, j] = o.get(f) or 0
            elif f in NUMERIC:
                v = o.get(f)
                X[i, j] = np.nan if v is None or (isinstance(v, float) and math.isnan(v)) else v
            elif f == "doc_complete":
                X[i, j] = 1.0 if o["completeness"] == "complete" else 0.0
            elif f == "doc_stub":
                X[i, j] = 1.0 if o["completeness"] == "stub" else 0.0
            elif f == "log_body":
                X[i, j] = math.log10(1 + o["body_len"])
            elif f == "log_links":
                X[i, j] = math.log10(1 + o["links"])
            elif f == "log_media":
                X[i, j] = math.log10(1 + o["media"])
            elif f == "fields_filled":
                X[i, j] = sum(1 for k in filled_fields if o.get(k)) / len(filled_fields)
    return pd.DataFrame(X, columns=[c[0] for c in cols])

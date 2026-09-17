"""
Canonical label definitions shared by the dataset pipeline (Phase 7) and the
model (Section 10), so the NER emission order, the B/I-to-entity-type
combination matrix, and the severity class order are defined exactly once.
"""

# Order matches the guide's entity table (1.2).
ENTITY_TYPES = [
    "Symptom",
    "Health Condition",
    "Medicine",
    "Age",
    "Dosage",
    "Specialist",
    "Medical Procedure",
]

# O + B/I for each of the 7 entity types = 15 labels (guide 1.2 / 10.2).
NER_LABELS = ["O"] + [f"B-{t}" for t in ENTITY_TYPES] + [f"I-{t}" for t in ENTITY_TYPES]
assert len(NER_LABELS) == 15

NER_LABEL2ID = {label: i for i, label in enumerate(NER_LABELS)}
NER_ID2LABEL = {i: label for label, i in NER_LABEL2ID.items()}

# Order matches the guide's stated severity head output order (10.3).
SEVERITY_LABELS = ["Emergency", "Urgent", "Routine", "General Query"]
SEVERITY_LABEL2ID = {label: i for i, label in enumerate(SEVERITY_LABELS)}
SEVERITY_ID2LABEL = {i: label for label, i in SEVERITY_LABEL2ID.items()}

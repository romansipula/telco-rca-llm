import json

from telco_rca.data.telecomts import KPI_SPECS, LABELS, parse_ticket
from telco_rca.schema import build_schema, dump_target, parse_output

SCHEMA = build_schema(LABELS, [k.name for k in KPI_SPECS])
GOOD = {"root_cause": "Jamming", "affected_kpis": ["DL_BLER", "UL_BLER"],
        "explanation": "Both BLERs rose sharply while RSRP stayed put.", "resolution": "Locate and isolate the jammer."}


def test_valid_output():
    p = parse_output(dump_target(GOOD), SCHEMA)
    assert p.json_ok and p.schema_ok and p.root_cause == "Jamming" and p.affected_kpis == ["DL_BLER", "UL_BLER"]


def test_code_fence_and_prose_are_tolerated_for_parsing_but_case_mismatch_is_not_schema_valid():
    txt = "Sure!\n```json\n" + json.dumps(dict(GOOD, root_cause="jamming")) + "\n```"
    p = parse_output(txt, SCHEMA)
    assert p.json_ok and not p.schema_ok and p.root_cause == "Jamming"


def test_invalid_outputs():
    assert not parse_output("no json here", SCHEMA).json_ok
    bad = dict(GOOD, root_cause="Alien attack")
    p = parse_output(json.dumps(bad), SCHEMA)
    assert p.json_ok and not p.schema_ok and p.root_cause is None
    p = parse_output(json.dumps(dict(GOOD, extra=1)), SCHEMA)
    assert not p.schema_ok
    p = parse_output(json.dumps({k: v for k, v in GOOD.items() if k != "resolution"}), SCHEMA)
    assert not p.schema_ok
    p = parse_output(json.dumps(dict(GOOD, affected_kpis=["NotAKPI"])), SCHEMA)
    assert not p.schema_ok and p.affected_kpis == []


def test_ticket_parser():
    t = parse_ticket("**Diagnose Summary:**\n- **Issue:** A.\n- **Symptoms:** B; C.\n- **Root Cause:** D.\n"
                     "- **Resolution:** E.")
    assert (t.issue, t.symptoms, t.root_cause, t.resolution) == ("A.", "B; C.", "D.", "E.")
    t = parse_ticket("**Diagnose Summary:** **Issue:** A. **Symptoms:** B. **Root Cause:** D. **Resolution:** E.")
    assert t.resolution == "E." and t.issue == "A."

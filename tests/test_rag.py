from srlab import rag


def test_context_and_check():
    ctx = rag.context("C07-m1-t1445714882")
    assert ctx and len(ctx["retrieved"]) == 3 and ctx["case_id"] == "C07-m1-t1445714882"
    p = rag.prompt(ctx)
    assert "[3]" in p and "F3" not in p and "F4" not in p  # plain-English fix names only
    ok = {"diagnosis": f"The ranker put it first at {ctx['current']['p0']}.", "precedent": "Same as [1] and [2].", "recommendation": "Try it.", "caveat": "None."}
    assert rag.check(ctx, ok) == []
    bad = {**ok, "precedent": "Same as [4], seen 1234 times."}
    assert ("precedent", "[4]") in rag.check(ctx, bad) and ("precedent", "1234") in rag.check(ctx, bad)
